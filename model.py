# model.py
import torch
import torch.nn as nn

def conv_block(in_c, out_c, k, bn=False):
    layers = [nn.Conv2d(in_c, out_c, kernel_size=k, padding=0), nn.ReLU(inplace=True)]
    if bn: layers.append(nn.BatchNorm2d(out_c))
    return nn.Sequential(*layers)

class TwoBranchCNN(nn.Module):
    """
    Branch1(3x3): 16,16 -> pool -> 64,64 -> pool -> 256 -> pool -> BN+Dropout -> FC128
    Branch2(5x5): 32,32 -> pool -> 128,128 -> pool -> 512 -> pool -> BN+Dropout -> FC128
    多头输出：
      - 'b1'   : 仅分支1的分类头
      - 'b2'   : 仅分支2的分类头
      - 'both' : 拼接后融合头（论文主结果）
    """
    def __init__(self, cfg, mode='both'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode  # 兼容旧接口；训练/评估建议直接用 forward() 返回的 dict
        d = float(getattr(cfg.model, "dropout", 0.2))
        self.num_classes = int(cfg.data.num_classes)

        # ---- Branch 1 (3x3) ----
        self.b1 = nn.Sequential(
            conv_block(3, 16, 3),
            conv_block(16, 16, 3),
            nn.MaxPool2d(2),
            conv_block(16, 64, 3),
            conv_block(64, 64, 3),
            nn.MaxPool2d(2),
            conv_block(64, 256, 3),
            nn.MaxPool2d(2),
        )
        self.b1_norm_drop = nn.Sequential(nn.BatchNorm2d(256), nn.Dropout(p=d))

        # ---- Branch 2 (5x5) ----
        self.b2 = nn.Sequential(
            conv_block(3, 32, 5),
            conv_block(32, 32, 5),
            nn.MaxPool2d(2),
            conv_block(32, 128, 5),
            conv_block(128, 128, 5),
            nn.MaxPool2d(2),
            conv_block(128, 512, 5),
            nn.MaxPool2d(2),
        )
        self.b2_norm_drop = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(p=d))

        # heavy-FC 路径（保留 H×W，lazy 推断 in_features）
        self.pool1 = nn.Identity()
        self.pool2 = nn.Identity()

        # 懒初始化的全连接与分类头
        self.fc1 = None          # Flatten -> Linear(in1,128) -> ReLU
        self.fc2 = None          # Flatten -> Linear(in2,128) -> ReLU
        self.head_b1 = None      # Linear(128,  num_classes)
        self.head_b2 = None      # Linear(128,  num_classes)
        self.head_both = None    # Linear(256,  num_classes)

    # ---------- 便捷操作：只训/冻结某一分支 ----------
    def freeze_branch(self, which: str):
        assert which in {'b1','b2'}
        for p in getattr(self, which).parameters(): p.requires_grad = False
        for p in getattr(self, f"{which}_norm_drop").parameters(): p.requires_grad = False

    def unfreeze_branch(self, which: str):
        assert which in {'b1','b2'}
        for p in getattr(self, which).parameters(): p.requires_grad = True
        for p in getattr(self, f"{which}_norm_drop").parameters(): p.requires_grad = True

    def set_mode(self, mode: str):
        # 兼容旧代码；实际 forward() 已返回全部头
        assert mode in {'both','b1','b2'}
        self.mode = mode

    # ---------- 内部：lazy 初始化 FC 与 分类头 ----------
    def _lazy_init_fc_and_heads(self, x1, x2):
        if self.fc1 is None:
            in1 = x1.view(x1.size(0), -1).size(1)
            in2 = x2.view(x2.size(0), -1).size(1)
            self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(in1, 128), nn.ReLU(inplace=True))
            self.fc2 = nn.Sequential(nn.Flatten(), nn.Linear(in2, 128), nn.ReLU(inplace=True))
            self.fc1.to(x1.device); self.fc2.to(x2.device)
        if self.head_b1 is None:
            self.head_b1   = nn.Linear(128, self.num_classes).to(x1.device)
            self.head_b2   = nn.Linear(128, self.num_classes).to(x2.device)
            self.head_both = nn.Linear(256, self.num_classes).to(x1.device)

    def forward(self, x):
        # feature maps
        x1 = self.b1(x); x1 = self.b1_norm_drop(x1); x1 = self.pool1(x1)
        x2 = self.b2(x); x2 = self.b2_norm_drop(x2); x2 = self.pool2(x2)

        # lazy init
        self._lazy_init_fc_and_heads(x1, x2)

        # per-branch 128-d features
        f1 = self.fc1(x1)  # [B,128]
        f2 = self.fc2(x2)  # [B,128]

        # three heads
        logits_b1  = self.head_b1(f1)
        logits_b2  = self.head_b2(f2)
        logits_both= self.head_both(torch.cat([f1, f2], dim=1))  # [B,256]

        # 返回 dict，便于多任务联合训练
        return {"b1": logits_b1, "b2": logits_b2, "both": logits_both}

    # 兼容：单模式前向（用于你现有的 evaluate_mode 快速接入）
    def forward_mode(self, x, mode: str):
        outs = self.forward(x)
        assert mode in outs, f"mode must be one of {list(outs.keys())}"
        return outs[mode]
