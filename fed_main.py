"""
fed_main.py

Federated Learning Entrypoint.

This module orchestrates the federated training lifecycle using the FedAvg algorithm.
It handles client data discovery, local training execution, model aggregation,
and global evaluation.

Prerequisites:
    - Data must be preprocessed via `data_preprocess.py` with `fed_mode=True`.
    - Directory structure:
        process_data/
            train/client_XXX/
            validation/
            test/
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Local application imports
# Assuming these modules exist and are importable
from model import TwoBranchCNN
from pipeline import (
    train_one_epoch,
    evaluate,
    evaluate_mode,
    plot_confusion,
    plot_history_curves,
)
from utils import (
    load_json_config,
    set_seed,
    make_run_dir,
    compute_class_weights_from_counts,
    get_device,
    fedavg,
    EarlyStopper,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class FederatedTrainer:
    """
    Manages the federated learning training lifecycle, including client management,
    aggregation, and evaluation.
    """

    def __init__(self, cfg: Any):
        """
        Initialize the FederatedTrainer.

        Args:
            cfg: Configuration object (DotDict or Namespace) containing parameters.
        """
        self.cfg = cfg
        self.device = get_device()
        self.run_dir: Path = Path(".")
        self.history: List[Dict[str, Any]] = []
        self.best_metrics = {"val_acc": 0.0}
        
        # Data containers
        self.client_loaders: Dict[int, DataLoader] = {}
        self.client_datasets: Dict[int, datasets.ImageFolder] = {}
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.class_names: List[str] = []
        
        # Setup
        self._setup_environment()
        self._setup_directories()
        
    def _setup_environment(self) -> None:
        """Sets random seeds and logs environment details."""
        set_seed(self.cfg.training.seed)
        logger.info(f"Initialized with seed: {self.cfg.training.seed}")
        logger.info(f"Compute device: {self.device}")

    def _setup_directories(self) -> None:
        """Creates necessary run directories for checkpoints and figures."""
        self.run_dir = make_run_dir(self.cfg.paths.save_dir, self.cfg.paths.run_naming)
        (self.run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "figs").mkdir(parents=True, exist_ok=True)
        logger.info(f"Run directory created at: {self.run_dir}")

    def _build_transforms(self) -> transforms.Compose:
        """Constructs the image preprocessing pipeline."""
        return transforms.Compose([
            transforms.Resize((self.cfg.data.img_size, self.cfg.data.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load_data(self) -> None:
        """
        Discovers and loads client, validation, and test datasets.
        
        Raises:
            RuntimeError: If no client directories are found.
        """
        tfm = self._build_transforms()
        data_dir = Path(self.cfg.paths.data_dir)
        train_root = data_dir / "train"
        
        # Discover clients
        client_paths = sorted(
            p for p in train_root.iterdir() 
            if p.is_dir() and p.name.startswith("client_")
        )

        if not client_paths:
            raise RuntimeError(
                f"No 'client_*' directories found in {train_root}. "
                "Ensure data_preprocess.py was run with fed_mode=True."
            )

        pin_memory = (self.device.type == "cuda")
        
        # Initialize Client Loaders
        for cid, cpath in enumerate(client_paths):
            ds = datasets.ImageFolder(cpath, transform=tfm)
            self.client_datasets[cid] = ds
            self.client_loaders[cid] = DataLoader(
                ds,
                batch_size=self.cfg.data.batch_size,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=pin_memory,
            )

        # Global Validation/Test Loaders
        val_ds = datasets.ImageFolder(data_dir / "validation", transform=tfm)
        test_ds = datasets.ImageFolder(data_dir / "test", transform=tfm)
        
        self.val_loader = DataLoader(
            val_ds, batch_size=self.cfg.data.batch_size, shuffle=False,
            num_workers=self.cfg.data.num_workers, pin_memory=pin_memory
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.cfg.data.batch_size, shuffle=False,
            num_workers=self.cfg.data.num_workers, pin_memory=pin_memory
        )
        
        self.class_names = val_ds.classes
        logger.info(f"Data Loaded: {len(self.client_loaders)} Clients, "
                    f"Classes: {len(self.class_names)}")

    def _compute_global_class_weights(self) -> Optional[torch.Tensor]:
        """Computes class weights across all clients to handle imbalance."""
        if not self.cfg.imbalance.use_class_weights:
            return None

        if self.cfg.imbalance.weights_mode == "manual":
            return compute_class_weights_from_counts(self.cfg.imbalance.manual_counts)

        # Auto aggregation
        counts = np.zeros(len(self.class_names), dtype=np.int64)
        for ds in self.client_datasets.values():
            # Handle both ImageFolder targets and generic lists
            targets = getattr(ds, "targets", [y for _, y in ds.samples])
            for y in targets:
                counts[y] += 1
        
        return compute_class_weights_from_counts(counts)

    def _initialize_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Initializes the global model and performs a dummy forward pass 
        to initialize lazy layers.
        """
        model = TwoBranchCNN(self.cfg).to(self.device)
        
        # Dummy forward to trigger lazy initialization of Linear layers
        dummy = torch.zeros(1, 3, self.cfg.data.img_size, self.cfg.data.img_size, device=self.device)
        with torch.no_grad():
            _ = model(dummy)
            
        return model, model.state_dict()

    def _train_single_client(
        self, 
        cid: int, 
        global_state: Dict[str, Any], 
        current_lr: float, 
        criterion: nn.Module,
        local_epochs: int,
        round_idx: int,
        best_client_tracker: Dict
    ) -> Tuple[Dict[str, Any], int, float, Dict]:
        """
        Performs local training on a specific client.
        """
        # Re-instantiate model to ensure clean state
        local_model = TwoBranchCNN(self.cfg).to(self.device)
        
        # Lazy init trigger
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.cfg.data.img_size, self.cfg.data.img_size, device=self.device)
            _ = local_model(dummy)
            
        local_model.load_state_dict(global_state)
        
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=current_lr,
            weight_decay=self.cfg.training.weight_decay
        )
        
        use_amp = (self.cfg.training.mixed_precision and self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        
        client_loader = self.client_loaders[cid]
        total_loss = 0.0
        last_metrics = {}
        
        local_model.train()
        for _ in range(local_epochs):
            epoch_loss, epoch_metrics = train_one_epoch(
                local_model, client_loader, optimizer, scaler, self.device, criterion
            )
            total_loss += epoch_loss
            last_metrics = epoch_metrics

        avg_loss = total_loss / max(local_epochs, 1)
        client_samples = len(client_loader.dataset)
        
        # Update client best tracking
        self._update_client_best(best_client_tracker, cid, round_idx, avg_loss, last_metrics)
        
        state_dict = local_model.state_dict()
        
        # Cleanup to save VRAM
        del local_model, optimizer, scaler
        torch.cuda.empty_cache()
        
        return state_dict, client_samples, avg_loss, last_metrics

    def _update_client_best(self, tracker: Dict, cid: int, round_idx: int, loss: float, metrics: Dict):
        """Updates the best historical metrics for a specific client."""
        if not metrics:
            return
        
        if metrics.get("acc", 0.0) > tracker[cid]["best_val_acc"]:
            tracker[cid].update({
                "best_round": round_idx,
                "best_val_loss": float(loss),
                "best_val_acc": float(metrics.get("acc", 0.0)),
                "best_val_f1": float(metrics.get("f1", 0.0)),
                "best_val_recall": float(metrics.get("recall", 0.0)),
                "best_val_mcc": float(metrics.get("mcc", 0.0)),
            })

    def run(self) -> Dict[str, Any]:
        """
        Executes the main federated training loop.
        """
        self.load_data()
        
        # Model & Optimization Setup
        global_model, global_state = self._initialize_model()
        
        class_weights = self._compute_global_class_weights()
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            logger.info(f"Class weights applied: {class_weights.tolist()}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Server optimizer handles LR scheduling, not parameter updates
        server_optimizer = torch.optim.Adam(
            global_model.parameters(), 
            lr=self.cfg.training.lr, 
            weight_decay=self.cfg.training.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            server_optimizer, mode="min", factor=0.5, patience=2
        )
        stopper = EarlyStopper(patience=self.cfg.training.patience, min_delta=0.0)

        # Federated Config
        fed_cfg = getattr(self.cfg, "fed_training", None)
        rounds = fed_cfg.rounds if fed_cfg else self.cfg.training.epochs
        local_epochs = fed_cfg.local_epochs if fed_cfg else 1
        frac_clients = fed_cfg.fraction_clients if fed_cfg else 1.0
        
        logger.info(f"Configuration: Rounds={rounds}, Local Epochs={local_epochs}, Client Fraction={frac_clients}")

        # Tracking per-client bests
        best_client_tracker = {
            cid: {"best_round": None, "best_val_acc": 0.0} for cid in self.client_loaders
        }

        # --- Main Round Loop ---
        for rnd in range(1, rounds + 1):
            t0 = time.time()
            
            # 1. Client Selection
            num_participating = max(1, int(frac_clients * len(self.client_loaders)))
            selected_clients = random.sample(list(self.client_loaders.keys()), num_participating)
            
            current_lr = server_optimizer.param_groups[0]["lr"]
            logger.info(f"Round {rnd}/{rounds} | LR: {current_lr:.6f} | Clients: {selected_clients}")

            client_states = []
            client_sizes = []
            round_train_losses = []
            round_train_metrics = []

            # 2. Local Training
            for cid in selected_clients:
                c_state, c_n, c_loss, c_metrics = self._train_single_client(
                    cid, global_state, current_lr, criterion, local_epochs, rnd, best_client_tracker
                )
                client_states.append(c_state)
                client_sizes.append(c_n)
                round_train_losses.append((c_loss, c_n))
                round_train_metrics.append((c_metrics, c_n))

            # 3. Aggregation (FedAvg)
            global_state = fedavg(global_state, client_states, client_sizes)
            global_model.load_state_dict(global_state)

            # 4. Metrics Aggregation
            train_loss = self._aggregate_metrics(round_train_losses)
            train_metrics = self._aggregate_metric_dicts(round_train_metrics)

            # 5. Global Evaluation
            val_loss, val_metrics = evaluate(global_model, self.val_loader, self.device, criterion)
            
            scheduler.step(val_loss)
            elapsed = time.time() - t0

            self._log_round(rnd, train_loss, train_metrics, val_loss, val_metrics, current_lr, elapsed)
            
            # 6. Checkpointing
            if val_metrics["acc"] > self.best_metrics["val_acc"]:
                self.best_metrics.update({
                    "epoch": rnd,
                    "val_acc": val_metrics["acc"],
                    "val_loss": val_loss,
                    **val_metrics # unpack f1, recall, etc.
                })
                save_checkpoint(self.run_dir / "ckpt" / "best.pt", global_model, server_optimizer, rnd)

            if stopper.should_stop(val_loss):
                logger.info(f"Early stopping triggered at round {rnd}")
                break

        # --- Finalization ---
        self._save_client_bests(best_client_tracker)
        results = self._final_evaluation(global_model, criterion)
        
        logger.info(f"Federated training complete. Results saved to {self.run_dir}")
        return results

    def _aggregate_metrics(self, loss_counts: List[Tuple[float, int]]) -> float:
        """Weighted average of losses."""
        total_samples = sum(n for _, n in loss_counts)
        if total_samples == 0: return 0.0
        return sum(l * n for l, n in loss_counts) / total_samples

    def _aggregate_metric_dicts(self, metrics_counts: List[Tuple[Dict, int]]) -> Dict[str, float]:
        """Weighted average of metric dictionaries."""
        total_samples = sum(n for _, n in metrics_counts)
        if total_samples == 0: return {"acc": 0.0}
        
        keys = metrics_counts[0][0].keys()
        agg = {k: 0.0 for k in keys}
        
        for m, n in metrics_counts:
            for k in keys:
                agg[k] += m[k] * n
                
        return {k: v / total_samples for k, v in agg.items()}

    def _log_round(self, rnd, train_loss, train_metric, val_loss, val_metric, lr, time_sec):
        """Logs round results to console and history file."""
        log_msg = (
            f"[Round {rnd:03d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_metric['acc']:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metric['acc']:.4f} | "
            f"Time: {time_sec:.1f}s"
        )
        logger.info(log_msg)

        row = {
            "epoch": rnd,
            "train_loss": train_loss,
            "train_acc": train_metric["acc"],
            "val_loss": val_loss,
            "val_acc": val_metric["acc"],
            "lr": lr,
            "time_sec": time_sec,
            # Add other metrics as needed
            **{f"train_{k}": v for k, v in train_metric.items() if k != "acc"},
            **{f"val_{k}": v for k, v in val_metric.items() if k != "acc"}
        }
        self.history.append(row)
        
        # Incremental save
        self._save_history()

    def _save_history(self):
        """Writes history to CSV and JSON."""
        csv_path = self.run_dir / "history.csv"
        if not self.history: return
        
        keys = self.history[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
            
        with open(self.run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def _save_client_bests(self, tracker: Dict):
        """Saves best metrics per client to a text file."""
        path = self.run_dir / "client_best_metrics.txt"
        with open(path, "w", encoding="utf-8") as f:
            for cid in sorted(tracker.keys()):
                b = tracker[cid]
                f.write(f"Client {cid}\n")
                if b["best_round"] is None:
                    f.write("  (no metrics recorded)\n\n")
                    continue
                for k, v in b.items():
                    if isinstance(v, float):
                        f.write(f"  {k:<15}: {v:.4f}\n")
                    else:
                        f.write(f"  {k:<15}: {v}\n")
                f.write("\n")

    def _final_evaluation(self, global_model: nn.Module, criterion: nn.Module) -> Dict:
        """Performs final evaluation on test set using best checkpoint."""
        logger.info("Starting final evaluation...")
        
        # Load best checkpoint
        ckpt_path = self.run_dir / "ckpt" / "best.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            global_model.load_state_dict(checkpoint["model"])
        else:
            logger.warning("Best checkpoint not found. Using current model state.")

        ftype = str(self.cfg.model.fuse_type).strip().lower()
        modes = [("CNN1", "b1"), ("CNN2", "b2"), ("Proposed", "both")] if ftype == "parallel" else [("Serial", "both")]
        
        overall_rows = []
        
        for label, mode in modes:
            overall, report_txt, cm, per_class = evaluate_mode(
                global_model, self.test_loader, self.device, criterion, self.class_names, mode
            )
            overall_rows.append({"model": label, **overall})
            
            # Save artifacts
            (self.run_dir / f"{label}_classification_report.txt").write_text(report_txt, encoding="utf-8")
            
            plot_confusion(cm, self.class_names, self.run_dir / "figs" / f"{label}_cm.png", normalize=False)
            plot_confusion(cm, self.class_names, self.run_dir / "figs" / f"{label}_cm_norm.png", normalize=True)

            # Save per-class metrics
            self._save_per_class_csv(label, per_class)

        # Save overall metrics
        keys = ["model", "loss", "acc", "recall", "precision", "balanced_acc", "mcc"]
        with open(self.run_dir / "overall_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(overall_rows)

        plot_history_curves(self.history, self.run_dir / "figs" / "train_val_curves.png")

        return {"run_dir": str(self.run_dir), "overall_results": overall_rows}

    def _save_per_class_csv(self, label: str, per_class_data: Dict):
        """Helper to save per-class metrics CSV."""
        path = self.run_dir / f"{label}_per_class_metrics.csv"
        headers = ["Class", "Precision", "Recall", "F1-score", "Support"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for c in self.class_names:
                d = per_class_data[c]
                w.writerow([c, f"{d['precision']:.2f}", f"{d['recall']:.2f}", 
                           f"{d['f1-score']:.2f}", int(d['support'])])


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Training Runner")
    parser.add_argument("--config", type=str, default="config.json", help="Path to JSON configuration file")
    return parser.parse_args()

def check_gpu():
    """Diagnostics for GPU availability."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU Detected: {device_name}")
        try:
            # Quick sanity check
            _ = torch.randn(2, 3, 64, 64, device="cuda")
            logger.info("CUDA Tensor test passed.")
        except Exception as e:
            logger.error(f"CUDA initialization failed: {e}")
    else:
        logger.warning("No GPU detected. Training will run on CPU (Slow).")

if __name__ == "__main__":
    args = parse_args()
    
    # 1. System Checks
    logger.info(f"Torch Version: {torch.__version__}")
    check_gpu()
    
    # 2. Load Config
    try:
        config = load_json_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file {args.config} not found.")
        sys.exit(1)
        
    # 3. Initialize and Run Trainer
    try:
        trainer = FederatedTrainer(config)
        trainer.run()
    except Exception as e:
        logger.critical(f"Unhandled exception during training: {e}", exc_info=True)
        sys.exit(1)