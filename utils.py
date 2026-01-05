"""
utils.py

Common utilities for configuration management, reproducibility, 
federated aggregation, and training helpers.
"""

import json
import logging
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Configure module-level logger
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in CuDNN backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to: {seed}")


def _dict_to_ns(d: Union[Dict, List, Any]) -> Union[SimpleNamespace, List, Any]:
    """
    Recursively converts a dictionary to a SimpleNamespace for dot-notation access.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_ns(x) for x in d]
    return d


def load_json_config(path: str = "config.json") -> SimpleNamespace:
    """
    Loads a JSON configuration file and resolves relative paths.

    Args:
        path (str): Relative path to the config file (default: "config.json").

    Returns:
        SimpleNamespace: Configuration object with resolved paths.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    root = Path(__file__).resolve().parent
    config_path = root / path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON config: {e}")

    ns = _dict_to_ns(cfg)

    # Resolve core paths
    ns.paths.root_dir = root
    ns.paths.data_dir = (root / ns.paths.data_dir).resolve()
    ns.paths.save_dir = (root / ns.paths.save_dir).resolve()
    
    return ns


def make_run_dir(base_dir: Path, mode: str = "datetime") -> Path:
    """
    Creates a unique directory for the current run.

    Args:
        base_dir (Path): The root directory for saving runs.
        mode (str): Naming strategy - 'datetime' (default) or 'increment'.

    Returns:
        Path: The created directory path.
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    if mode == "increment":
        i = 1
        while (base_dir / f"{i:04d}").exists():
            i += 1
        run_dir = base_dir / f"{i:04d}"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = base_dir / timestamp

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"Run directory created: {run_dir}")
    except FileExistsError:
        # Fallback for millisecond collisions in datetime mode
        run_dir = run_dir.with_name(f"{run_dir.name}_{random.randint(0,999)}")
        run_dir.mkdir()
        logger.warning(f"Run directory collision resolved: {run_dir}")

    return run_dir


def compute_class_weights_from_counts(counts: List[int]) -> torch.Tensor:
    """
    Computes inverse class frequencies as weights.

    Formula: w_j = N / (C * n_j)
    
    Args:
        counts (List[int]): List of sample counts per class.

    Returns:
        torch.Tensor: Float tensor of weights.
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    total_samples = counts_arr.sum()
    num_classes = len(counts_arr)

    # Avoid division by zero for classes with 0 samples
    safe_counts = np.where(counts_arr == 0, 1.0, counts_arr)
    
    weights = total_samples / (num_classes * safe_counts)
    
    # Zero out weights for empty classes to prevent bad updates
    weights[counts_arr == 0] = 0.0
    
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights_from_dataset(dataset: Any) -> Tuple[torch.Tensor, List[int]]:
    """
    Extracts targets from a dataset and calculates class weights.

    Args:
        dataset: PyTorch Dataset object (must have .targets or .samples).

    Returns:
        Tuple[torch.Tensor, List[int]]: (Computed weights, raw counts)
    """
    # Attempt to fetch targets efficiently
    targets = getattr(dataset, "targets", None)
    if targets is None and hasattr(dataset, "samples"):
        # Fallback for ImageFolder if .targets isn't cached
        targets = [y for _, y in dataset.samples]
    
    if targets is None:
        raise ValueError("Dataset does not expose 'targets' or 'samples'.")

    # Count frequencies
    counter = Counter(targets)
    # Ensure ordered list matching class indices 0..C-1
    num_classes = len(getattr(dataset, "classes", [])) or (max(targets) + 1)
    counts = [counter[i] for i in range(num_classes)]

    weights = compute_class_weights_from_counts(counts)
    return weights, counts


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    """
    Saves the model state, optimizer state, and current epoch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(state, path)
    logger.debug(f"Checkpoint saved: {path}")


def get_device() -> torch.device:
    """Returns the appropriate torch device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def fedavg(
    global_state: Dict[str, torch.Tensor],
    client_states: List[Dict[str, torch.Tensor]],
    client_sizes: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Performs Federated Averaging (FedAvg) on model state dictionaries.

    Logic:
        - Floating point parameters: Weighted average based on client sample size.
        - Non-floating buffers (e.g. LongTensor, IntTensor): Copied from the first client.
          (Averaging integers is usually invalid for flags or counters).

    Args:
        global_state: The state_dict of the global model.
        client_states: List of state_dicts from local clients.
        client_sizes: List of integers representing the number of samples per client.

    Returns:
        Dict[str, torch.Tensor]: The aggregated global state dict.
    """
    if not client_states:
        logger.warning("fedavg called with no client states. Returning global state unchanged.")
        return global_state

    total_samples = sum(client_sizes)
    if total_samples == 0:
        logger.warning("Total samples in fedavg is 0. Returning global state.")
        return global_state

    # Initialize the new state
    new_state: Dict[str, torch.Tensor] = {}

    # Initialize accumulators for floating point params
    # Non-float params (e.g. num_batches_tracked) are copied from the first client explicitly
    for k, v in global_state.items():
        if torch.is_floating_point(v):
            new_state[k] = torch.zeros_like(v)
        else:
            # Default to the first client's value for non-aggregatable buffers
            # This ensures we don't accidentally keep the stale global value if clients updated it
            new_state[k] = client_states[0][k].clone()

    # Aggregate
    for state, size in zip(client_states, client_sizes):
        weight = float(size) / total_samples
        
        for k, v in state.items():
            if k not in new_state:
                continue # Skip keys that might be missing in global/local mismatch scenarios

            if torch.is_floating_point(new_state[k]):
                # Accumulate weighted parameters
                # .to() ensures device/dtype consistency
                new_state[k] += weight * v.to(new_state[k].dtype)

    return new_state


class EarlyStopper:
    """
    Implements early stopping logic to halt training when validation metric stops improving.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float("inf")
        self.early_stop = False

    def should_stop(self, val_loss: float) -> bool:
        """
        Checks if training should stop.
        
        Args:
            val_loss (float): Current validation loss.
            
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False