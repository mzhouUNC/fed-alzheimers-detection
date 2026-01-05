from utils import load_json_config
from pipeline import fit
import torch, torchvision, os, sys

if __name__ == "__main__":
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        a=torch.randn(2,3,224,224, device="cuda"); b=torch.nn.Conv2d(3,8,3).cuda()(a); print("ok:", b.shape)
    cfg = load_json_config("config.json")
    res = fit(cfg)
    print(res)
