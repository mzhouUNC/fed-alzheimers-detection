"""
data_preprocess.py

Data Preprocessing & Federated Splitting Module.

This script resizes images and splits them into Train/Validation/Test sets.
It supports advanced Federated Learning splitting strategies:
1. IID (Independent and Identically Distributed)
2. Non-IID (Dirichlet Label Skew)
3. Non-IID (Quantity Skew)

Usage:
    python data_preprocess.py --fed_mode True --fed_split dirichlet --fed_dirichlet_alpha 0.5
"""

import argparse
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Logging & Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]


class DataPreprocessor:
    """
    Handles image resizing, dataset partitioning, and federated data distribution.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path.cwd()
        self.src_root = self.root / "origin_data" / "AugmentedAlzheimerDataset"
        self.dst_root = self.root / "process_data"
        
        # State for Federated Splitting
        self.qty_probs: Optional[List[float]] = None
        self.qty_rng: Optional[random.Random] = None
        self.dirichlet_rngs: Dict[str, np.random.Generator] = {}
        self.dirichlet_probs: Dict[str, np.ndarray] = {}

        self._validate_environment()
        self._setup_federated_strategies()

    def _validate_environment(self) -> None:
        """Checks directory structure and ensures cleanliness."""
        if not self.src_root.exists():
            logger.error(f"Source directory not found: {self.src_root}")
            sys.exit(1)

        # Check for origin_data folder in root to prevent running in wrong dir
        if not (self.root / "origin_data").exists():
            logger.error(f"Please run from the project root containing 'origin_data'. Current: {self.root}")
            sys.exit(1)

        if self.dst_root.exists():
            logger.info(f"Removing existing output directory: {self.dst_root}")
            shutil.rmtree(self.dst_root)

        # Create base structure
        for split in ["train", "validation", "test"]:
            (self.dst_root / split).mkdir(parents=True, exist_ok=True)

    def _setup_federated_strategies(self) -> None:
        """Pre-calculates probability distributions for Non-IID scenarios."""
        if not self.args.fed_mode:
            return

        mode = self.args.fed_split.lower()
        
        # Strategy: Quantity Skew (Different clients get different total amounts of data)
        if mode == "quantity":
            logger.info("Initializing Quantity Skew distribution...")
            self.qty_rng = random.Random(self.args.seed + 999)
            # Example distribution: Arithmetic progression (Client 0 gets most, Client N gets least)
            weights = [self.args.num_clients - j for j in range(self.args.num_clients)]
            total_w = sum(weights)
            self.qty_probs = [w / total_w for w in weights]
            logger.info(f"Client Quantity Probabilities: {[round(p, 4) for p in self.qty_probs]}")

        # Strategy: Dirichlet (Label Skew - Pre-calculated per class later)
        elif mode == "dirichlet":
            logger.info(f"Initializing Dirichlet Label Skew (alpha={self.args.fed_dirichlet_alpha})...")
            # We initialize generators here, but specific probs are calculated per-class
            pass

    def _collect_files(self) -> Dict[str, List[Path]]:
        """Scans source directory for images."""
        pool = {}
        for cls in DEFAULT_CLASSES:
            cls_dir = self.src_root / cls
            if not cls_dir.is_dir():
                logger.error(f"Missing class directory: {cls_dir}")
                sys.exit(1)
            
            files = sorted([
                p for p in cls_dir.rglob("*") 
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            ])
            
            if not files:
                logger.error(f"No images found for class: {cls}")
                sys.exit(1)
            
            pool[cls] = files
        return pool

    def _split_indices(self, total: int) -> Tuple[Set[int], Set[int], Set[int]]:
        """Splits indices into Train/Val/Test sets."""
        ids = list(range(total))
        rnd = random.Random(self.args.seed)
        rnd.shuffle(ids)
        
        n_train = int(total * self.args.train)
        n_val = int(total * self.args.val)
        
        return set(ids[:n_train]), set(ids[n_train : n_train + n_val]), set(ids[n_train + n_val:])

    def _get_client_id(self, idx: int, cls: str) -> int:
        """Determines which client receives a training sample based on strategy."""
        mode = self.args.fed_split.lower()

        if mode == "iid":
            return idx % self.args.num_clients

        elif mode == "dirichlet":
            # Lazy init for specific class probabilities
            if cls not in self.dirichlet_probs:
                # Use class-specific seed for reproducibility
                cls_idx = DEFAULT_CLASSES.index(cls)
                rng = np.random.default_rng(self.args.seed + cls_idx)
                alpha_vec = np.full(self.args.num_clients, self.args.fed_dirichlet_alpha)
                probs = rng.dirichlet(alpha_vec)
                
                self.dirichlet_rngs[cls] = rng
                self.dirichlet_probs[cls] = probs
                logger.info(f"Class '{cls}' Dirichlet Distribution: {np.round(probs, 3)}")

            # Sample client based on class distribution
            return self.dirichlet_rngs[cls].choice(
                self.args.num_clients, p=self.dirichlet_probs[cls]
            )

        elif mode == "quantity":
            # Sample client based on global quantity distribution
            return self._sample_from_probs(self.qty_rng, self.qty_probs)

        else:
            raise ValueError(f"Unknown split mode: {mode}")

    @staticmethod
    def _sample_from_probs(rng: random.Random, probs: List[float]) -> int:
        """Helper to sample index from discrete probability distribution."""
        u = rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if u <= cumulative:
                return i
        return len(probs) - 1

    def _resize_and_save(self, src: Path, dst: Path) -> None:
        """Resizes image and saves as JPEG."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(src) as im:
                im = im.convert("RGB").resize(
                    (self.args.size, self.args.size), Image.Resampling.BILINEAR
                )
                # Change extension to .jpg
                dst_with_ext = dst.with_suffix(".jpg")
                im.save(dst_with_ext, format="JPEG", quality=95, optimize=True)
        except Exception as e:
            logger.warning(f"Failed to process {src.name}: {e}")

    def process(self):
        """Main processing pipeline."""
        pool = self._collect_files()
        total_files = sum(len(v) for v in pool.values())
        
        stats = defaultdict(int)  # Written counts per split
        client_stats = defaultdict(lambda: defaultdict(int)) # [client][class] -> count

        with tqdm(total=total_files, desc="Processing", unit="img") as pbar_total:
            for cls in tqdm(DEFAULT_CLASSES, desc="Classes", leave=False):
                files = pool[cls]

                # Subset logic
                if 0.0 < self.args.subset_ratio < 1.0:
                    rnd = random.Random(self.args.seed)
                    k = max(1, int(len(files) * self.args.subset_ratio))
                    files = rnd.sample(files, k)

                tr_ids, va_ids, te_ids = self._split_indices(len(files))

                for i, file_path in enumerate(files):
                    # Determine split
                    if i in tr_ids:
                        split = "train"
                    elif i in va_ids:
                        split = "validation"
                    else:
                        split = "test"

                    out_name = f"{cls}_{i:07d}" # Extension added in save
                    
                    # Federated Split Logic (Only affects training set)
                    if split == "train" and self.args.fed_mode:
                        cid = self._get_client_id(i, cls)
                        # Path: process_data/train/client_000/ClassName/Image.jpg
                        dst_path = self.dst_root / "train" / f"client_{cid:03d}" / cls / out_name
                        client_stats[cid][cls] += 1
                    else:
                        # Path: process_data/split/ClassName/Image.jpg
                        dst_path = self.dst_root / split / cls / out_name

                    self._resize_and_save(file_path, dst_path)
                    stats[split] += 1
                    pbar_total.update(1)

        self._print_summary(stats, client_stats)

    def _print_summary(self, stats: Dict, client_stats: Dict):
        print("\n" + "="*30)
        print(" GLOBAL DATASET SUMMARY")
        print("="*30)
        for k, v in stats.items():
            print(f" {k:<15}: {v}")

        if self.args.fed_mode:
            print("\n" + "="*30)
            print(" FEDERATED CLIENT DISTRIBUTION")
            print("="*30)
            total_samples = 0
            sorted_cids = sorted(client_stats.keys())
            
            # Ensure all clients are printed even if empty
            for i in range(self.args.num_clients):
                if i not in sorted_cids:
                    sorted_cids.append(i)
            sorted_cids.sort()

            for cid in sorted_cids:
                c_total = sum(client_stats[cid].values())
                total_samples += c_total
                print(f"\n[Client {cid:03d}] Total: {c_total}")
                for cls in DEFAULT_CLASSES:
                    count = client_stats[cid].get(cls, 0)
                    print(f"  - {cls:<16}: {count}")

        print("\nProcessing Complete.")
        print(f"Output Directory: {self.dst_root}")


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    return False

def load_defaults_from_config(root_dir: Path) -> Dict:
    """Reads default values from config.json if available."""
    cfg_path = root_dir / "config.json"
    defaults = {
        "subset_ratio": 1.0,
        "fed_mode": False,
        "num_clients": 1,
        "fed_split": "iid",
        "fed_dirichlet_alpha": 0.5
    }
    
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                data_cfg = cfg.get("data", {})
                defaults.update({
                    "subset_ratio": data_cfg.get("subset_ratio", 1.0),
                    "fed_mode": data_cfg.get("fed_mode", False),
                    "num_clients": data_cfg.get("num_clients", 1),
                    "fed_split": data_cfg.get("fed_split", "iid"),
                    "fed_dirichlet_alpha": data_cfg.get("fed_dirichlet_alpha", 0.5),
                })
        except Exception as e:
            logger.warning(f"Failed to read config.json: {e}. Using hardcoded defaults.")
    
    return defaults

def main():
    root = Path.cwd()
    defaults = load_defaults_from_config(root)

    parser = argparse.ArgumentParser(description="Image Preprocessing & Federated Splitter")
    
    # Standard Params
    parser.add_argument("--size", type=int, default=224, help="Target image size (WxH)")
    parser.add_argument("--train", type=float, default=0.81, help="Train split ratio")
    parser.add_argument("--val", type=float, default=0.09, help="Validation split ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset_ratio", type=float, default=defaults["subset_ratio"], 
                        help="Ratio of data to use (for debugging)")

    # Federated Params
    parser.add_argument("--fed_mode", type=str2bool, default=defaults["fed_mode"],
                        help="Enable federated client splitting")
    parser.add_argument("--num_clients", type=int, default=defaults["num_clients"],
                        help="Number of federated clients")
    parser.add_argument("--fed_split", type=str, default=defaults["fed_split"],
                        choices=["iid", "quantity", "dirichlet"],
                        help="Splitting strategy: 'iid', 'quantity' (skewed amounts), 'dirichlet' (skewed labels)")
    parser.add_argument("--fed_dirichlet_alpha", type=float, default=defaults["fed_dirichlet_alpha"],
                        help="Concentration parameter for Dirichlet distribution (Lower = more skewed)")

    args = parser.parse_args()

    # Validation
    if abs(args.train + args.val + args.test - 1.0) > 1e-5:
        logger.error("Train/Val/Test ratios must sum to 1.0")
        sys.exit(1)

    if args.fed_mode and args.num_clients < 1:
        logger.error("num_clients must be >= 1 when fed_mode is active.")
        sys.exit(1)

    # Execution
    preprocessor = DataPreprocessor(args)
    preprocessor.process()

if __name__ == "__main__":
    main()