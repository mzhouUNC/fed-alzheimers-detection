import json, time
import csv
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix, recall_score, matthews_corrcoef, precision_score, balanced_accuracy_score

from utils import (
    set_seed, EarlyStopper, save_checkpoint, make_run_dir,
    compute_class_weights_from_dataset, compute_class_weights_from_counts
)
from model import TwoBranchCNN

def summarize_overall(y_true, y_pred, loss):
    """生成表1需要的总体指标"""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    rec = float(recall_score(y_true, y_pred, average="macro"))
    pre = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    return {"loss": float(loss), "acc": acc, "recall": rec, "precision": pre, "balanced_acc": bal, "mcc": mcc}


def evaluate_mode(model, loader, device, criterion, class_names, mode: str):
    """一次评估某个分支/组合；返回总体表与分类报告、混淆矩阵"""
    assert mode in {"b1","b2","both"}
    model.eval()  # 不再依赖 set_mode，直接从 dict 里取
    loss_sum, total = 0.0, 0
    y_true, y_pred = [], []
    to_cuda = (device.type == "cuda")

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=to_cuda)
            y = y.to(device, non_blocking=to_cuda, dtype=torch.long)
            outs   = model(x)             # dict: {"b1","b2","both"}
            logits = outs[mode]           # 关键改动
            loss   = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            total    += y.size(0)
            y_true.extend(y.tolist())
            y_pred.extend(logits.argmax(1).tolist())

    avg_loss = loss_sum / total
    overall = summarize_overall(y_true, y_pred, avg_loss)
    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_cls = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    per_class_table = {c: {k: float(v) for k, v in per_cls[c].items()} for c in class_names}
    return overall, report_txt, cm, per_class_table

def plot_history_curves(history, out_png):
    """history 来自你已有的记录，画 train/val loss 与 acc"""
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    tr_acc  = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.plot(epochs, tr_loss, label="train"); plt.plot(epochs, val_loss, label="val")
    plt.title("Loss"); plt.xlabel("epoch"); plt.legend()
    plt.subplot(1,2,2); plt.plot(epochs, tr_acc, label="train"); plt.plot(epochs, val_acc, label="val")
    plt.title("Accuracy"); plt.xlabel("epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def _compute_epoch_metrics(y_true, y_pred, average="macro"):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    f1  = float(f1_score(y_true, y_pred, average=average))
    rec = float(recall_score(y_true, y_pred, average=average))
    mcc = float(matthews_corrcoef(y_true, y_pred))  # sklearn 支持多分类
    return {"acc": acc, "f1": f1, "recall": rec, "mcc": mcc}

def make_loaders(cfg):
    tfm = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_root = cfg.paths.data_dir / "train"
    val_root   = cfg.paths.data_dir / "validation"
    test_root  = cfg.paths.data_dir / "test"

    train_ds = datasets.ImageFolder(train_root, transform=tfm)
    val_ds   = datasets.ImageFolder(val_root,   transform=tfm)
    test_ds  = datasets.ImageFolder(test_root,  transform=tfm)

    class_weights, sampler = None, None
    if cfg.imbalance.use_class_weights or cfg.imbalance.use_weighted_sampler:
        if cfg.imbalance.weights_mode == "manual":
            class_weights = compute_class_weights_from_counts(cfg.imbalance.manual_counts)
        else:
            class_weights, _ = compute_class_weights_from_dataset(train_ds)
        if cfg.imbalance.use_weighted_sampler:
            targets = getattr(train_ds, "targets", [y for _, y in train_ds.samples])
            sample_weights = class_weights[torch.tensor(targets, dtype=torch.long)]
            sampler = WeightedRandomSampler(weights=sample_weights.double(),
                                            num_samples=len(sample_weights),
                                            replacement=True)

    pin = (torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=cfg.data.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False,
                              num_workers=cfg.data.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False,
                              num_workers=cfg.data.num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader, class_weights

def plot_confusion(cm, classes, path_png, normalize=False, title=None):
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title or ("Confusion Matrix (norm)" if normalize else "Confusion Matrix"))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2. if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.savefig(path_png, dpi=200); plt.close()

@torch.no_grad()
def evaluate_full(model, loader, device, criterion, class_names):
    model.eval()
    loss_sum, total = 0.0, 0
    y_true, y_pred = [], []
    pbar = tqdm(loader, desc="eval", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        outs   = model(x)
        logits = outs["both"]          # 关键改动
        loss   = criterion(logits, y)
        loss_sum += loss.item() * x.size(0); total += y.size(0)
        pred = logits.argmax(1)
        y_true.extend(y.tolist()); y_pred.extend(pred.tolist())
        pbar.set_postfix(loss=loss_sum/total)

    avg_loss = loss_sum/total
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {
        "loss": avg_loss, "acc": acc,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "per_class_f1": {cls: f for cls, f in zip(class_names, per_class_f1)},
        "report_text": report, "cm": cm
    }


def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    running_loss, total = 0.0, 0
    y_true_all, y_pred_all = [], []
    pbar = tqdm(loader, desc="train", leave=False)

    # 可从 cfg 里取加权，没配就用默认
    w_all = getattr(getattr(model, "cfg", object()), "training", object())
    w_both = getattr(w_all, "w_both", 1.0) if hasattr(w_all, "w_both") else 1.0
    w_branch = getattr(w_all, "w_branch", 0.3) if hasattr(w_all, "w_branch") else 0.3

    for x, y in pbar:
        to_cuda = (device.type == "cuda")
        x = x.to(device, non_blocking=to_cuda)
        y = y.to(device, non_blocking=to_cuda, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outs    = model(x)                     # dict
                loss_b1 = criterion(outs["b1"],   y)
                loss_b2 = criterion(outs["b2"],   y)
                loss_bt = criterion(outs["both"], y)
                loss = w_both*loss_bt + w_branch*(loss_b1 + loss_b2)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            outs    = model(x)
            loss_b1 = criterion(outs["b1"],   y)
            loss_b2 = criterion(outs["b2"],   y)
            loss_bt = criterion(outs["both"], y)
            loss = w_both*loss_bt + w_branch*(loss_b1 + loss_b2)
            loss.backward(); optimizer.step()

        running_loss += loss.item() * x.size(0); total += y.size(0)

        # 即时 acc 仍用融合头（与你验证/选 best 保持一致）
        pred = outs["both"].argmax(1)
        y_true_all.extend(y.tolist()); y_pred_all.extend(pred.tolist())
        running_acc = float((np.array(y_true_all) == np.array(y_pred_all)).mean())
        pbar.set_postfix(loss=running_loss/total, acc=running_acc)

    epoch_loss = running_loss/total
    metrics = _compute_epoch_metrics(y_true_all, y_pred_all, average="macro")
    return epoch_loss, metrics


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, total = 0.0, 0
    y_true_all, y_pred_all = [], []
    pbar = tqdm(loader, desc="val", leave=False)

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        outs   = model(x)           # dict
        logits = outs["both"]       # 关键改动：验证只看融合头
        loss   = criterion(logits, y)

        loss_sum += loss.item() * x.size(0); total += y.size(0)
        pred = logits.argmax(1)
        y_true_all.extend(y.tolist()); y_pred_all.extend(pred.tolist())
        running_acc = float((np.array(y_true_all) == np.array(y_pred_all)).mean())
        pbar.set_postfix(loss=loss_sum/total, acc=running_acc)

    epoch_loss = loss_sum/total
    metrics = _compute_epoch_metrics(y_true_all, y_pred_all, average="macro")
    return epoch_loss, metrics


def fit(cfg) -> Dict:
    set_seed(cfg.training.seed)
    run_dir = make_run_dir(cfg.paths.save_dir, cfg.paths.run_naming)
    (run_dir/"ckpt").mkdir(parents=True, exist_ok=True)
    (run_dir/"figs").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    train_loader, val_loader, test_loader, class_weights = make_loaders(cfg)
    class_names = train_loader.dataset.classes

    model = TwoBranchCNN(cfg).to(device)
    if class_weights is not None and cfg.imbalance.use_class_weights:
        class_weights = class_weights.to(device)
        print("Class weights:", class_weights.tolist())

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr,
                                 weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    use_amp = (cfg.training.mixed_precision and device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    #scaler = torch.cuda.amp.GradScaler() if (cfg.training.mixed_precision and device.type=="cuda") else None
    stopper = EarlyStopper(patience=cfg.training.patience, min_delta=0.0)

    # 训练过程历史：每轮都会append
    history = []  # list[dict]

    best = {"val_acc": 0.0}
    for epoch in range(1, cfg.training.epochs+1):
        t0 = time.time()

        tr_loss, tr_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
        val_loss, val_metrics = evaluate(model, val_loader, device, criterion)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        # 打印本轮（包含 acc / F1 / recall / MCC）
        print(
            f"[{epoch:03d}] "
            f"train loss {tr_loss:.4f} | acc {tr_metrics['acc']:.4f} f1 {tr_metrics['f1']:.4f} "
            f"recall {tr_metrics['recall']:.4f} mcc {tr_metrics['mcc']:.4f} || "
            f"val loss {val_loss:.4f} | acc {val_metrics['acc']:.4f} f1 {val_metrics['f1']:.4f} "
            f"recall {val_metrics['recall']:.4f} mcc {val_metrics['mcc']:.4f} | {elapsed:.1f}s"
        )

        # 记录并落盘（每轮追加一行）
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_metrics["acc"],
            "train_f1": tr_metrics["f1"],
            "train_recall": tr_metrics["recall"],
            "train_mcc": tr_metrics["mcc"],
            "val_loss": val_loss,
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_mcc": val_metrics["mcc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": elapsed,
        }
        history.append(row)

        # 维护 best（以 val_acc 作为准则，若你想按 F1 改这里即可）
        if val_metrics["acc"] > best["val_acc"]:
            best.update(epoch=epoch, val_acc=val_metrics["acc"], val_loss=val_loss,
                        val_f1=val_metrics["f1"], val_recall=val_metrics["recall"], val_mcc=val_metrics["mcc"])
            save_checkpoint(run_dir/"ckpt"/"best.pt", model, optimizer, epoch)

        if stopper.should_stop(val_loss):
            print(f"Early stopped at epoch {epoch}")
            break

        # 将 history 追加写入 CSV（安全起见每轮都覆盖写一份最新的）
        import csv
        csv_path = run_dir/"history.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader(); writer.writerows(history)

        # JSON 也同步一份
        with open(run_dir/"history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    # ===== 测试 + 完整指标：b1 / b2 / both =====
    ckpt = torch.load(run_dir/"ckpt"/"best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    class_names = train_loader.dataset.classes

    ftype = str(cfg.model.fuse_type).strip().lower()
    modes = [("CNN1","b1"), ("CNN2","b2"), ("Proposed","both")] if ftype=="parallel" else [("Serial","both")]
    print("Test modes:", modes)
    overall_rows = []

    for label, mode in modes:
        overall, report_txt, cm, per_class = evaluate_mode(
            model, test_loader, device, criterion, class_names, mode
        )

        # 表1汇总
        overall_rows.append({"model": label, **overall})

        # 分类报告
        with open(run_dir / f"{label}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report_txt)

        # 混淆矩阵（原始+归一化）
        plot_confusion(cm, class_names, run_dir/"figs"/f"{label}_cm.png",
                    normalize=False, title=f"{label} Confusion Matrix")
        plot_confusion(cm, class_names, run_dir/"figs"/f"{label}_cm_norm.png",
                    normalize=True, title=f"{label} Confusion Matrix (Normalized)")

        # 每类指标 CSV
        head = ["Class","Precision","Recall","F1-score","Support"]
        with open(run_dir/f"{label}_per_class_metrics.csv", "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(head)
            for c in class_names:
                row = per_class[c]
                w.writerow([c, f"{row['precision']:.2f}", f"{row['recall']:.2f}",
                            f"{row['f1-score']:.2f}", int(row['support'])])

    # 写出 overall_metrics（含三行 CNN1/CNN2/Proposed）
    head = ["model","loss","acc","recall","precision","balanced_acc","mcc"]
    with open(run_dir/"overall_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=head); w.writeheader()
        for r in overall_rows: w.writerow(r)


    # 写出“表1”（与论文一致的列）
    import csv
    head = ["model","loss","acc","recall","precision","balanced_acc","mcc"]
    with open(run_dir/"overall_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=head); w.writeheader()
        for r in overall_rows: w.writerow(r)

    # 始终导出训练曲线（loss/acc）
    plot_history_curves(history, run_dir/"figs"/"train_val_curves.png")

    # 最后再写总体 JSON
    out = {
        "class_names": class_names,
        "best_epoch": best.get("epoch", None),
        "best_val_acc": best.get("val_acc", None),
        "best_val_loss": best.get("val_loss", None),
        "best_val_f1": best.get("val_f1", None),
        "best_val_recall": best.get("val_recall", None),
        "best_val_mcc": best.get("val_mcc", None),
        "overall_rows": overall_rows,
    }
    with open(run_dir/"metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {run_dir}")
    return {"run_dir": str(run_dir), **out}


   