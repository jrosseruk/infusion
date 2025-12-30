from typing import Any, Dict, Optional, Tuple, Literal
from torch.utils.data import Dataset

ReturnMode = Literal["pair", "infused", "original"]

class InfusableDataset(Dataset):
    """
    Overlay that can replace items at chosen indices.
    - Starts empty: infused == original
    - Later: set replacement items for some indices

    return_mode:
      - "pair":    returns (original, infused, idx)
      - "infused": returns (infused, idx) only (useful for normal training)
      - "original": returns (original, idx) only
    """
    def __init__(self, base: Dataset, return_mode: ReturnMode = "pair"):
        self.base = base
        self.return_mode = return_mode
        self._overlay: Dict[int, Any] = {}  # idx -> replacement item

    def set_return_mode(self, mode: ReturnMode):
        """Change the return mode after instantiation."""
        self.return_mode = mode

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        original = self.base[idx]
        infused = self._overlay.get(int(idx), original)

        if self.return_mode == "pair":
            return original, infused, idx
        if self.return_mode == "infused":
            return infused, idx
        if self.return_mode == "original":
            return original, idx
        raise ValueError(f"Unknown return_mode={self.return_mode}")

    # --- poisoning API ---
    def infuse_one(self, idx: int, replacement_item: Any) -> None:
        self._overlay[int(idx)] = replacement_item

    def infuse(self, updates: Dict[int, Any]) -> None:
        for k, v in updates.items():
            self._overlay[int(k)] = v

    def clear(self, idx: Optional[int] = None) -> None:
        if idx is None:
            self._overlay.clear()
        else:
            self._overlay.pop(int(idx), None)

    def is_infused(self, idx: int) -> bool:
        return int(idx) in self._overlay

    def num_infused(self) -> int:
        return len(self._overlay)


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    from torchvision import transforms
    import random

    # Define parameters
    NUM_POISON = 200
    RANDOM_SEED = 42

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)

    mnist = MNIST(root=".", train=True, download=True, transform=transforms.ToTensor())
    poisoned = InfusableDataset(mnist, return_mode="infused")  # train on poisoned view

    # Randomly select indices to poison
    total_samples = len(mnist)
    poison_indices = random.sample(range(total_samples), NUM_POISON)

    # Prepare poisoned replacements
    updates = {}
    for idx in poison_indices:
        x, y = mnist[idx]
        # Example: flip label
        if isinstance(y, int):
            y_poison = (y + 1) % 10
        else:
            # In rare cases, y could be a tensor; handle accordingly
            y_poison = ((y.item() + 1) % 10) if hasattr(y, 'item') else (int(y) + 1) % 10
        updates[idx] = (x, y_poison)  # full replacement

    poisoned.infuse(updates)
    print(f"Infused {len(poison_indices)} samples into the dataset.")
    print(f"Indices of infused examples: {sorted(poison_indices)}")

    # --- Demonstrate return_mode changes without recreating the dataset ---
    print("\n--- Demo: Changing return_mode and returned content ---")
    for mode in ["pair", "infused", "original"]:
        poisoned.set_return_mode(mode)
        print(f"\nReturn mode: {mode}")
        sample = poisoned[poison_indices[0]]
        print(f"Sample return (showing shapes/types):")
        if mode == "pair":
            (orig, inf, idx) = sample
            print(f"[pair] idx={idx}, original label={orig[1]}, infused label={inf[1]}")
        elif mode == "infused":
            (item, idx) = sample
            print(f"[infused] idx={idx}, label={item[1]}")
        elif mode == "original":
            (item, idx) = sample
            print(f"[original] idx={idx}, label={item[1]}")

