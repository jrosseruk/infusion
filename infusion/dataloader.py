from torch.utils.data import Dataset, DataLoader, Sampler
import torch



class FixedPermutationSampler(Sampler[int]):
    """
    Yields a single random permutation (seeded) of indices repeatedly.
    That means every epoch sees the same shuffled order.
    """
    def __init__(self, data_source: Dataset, seed: int = None):
        self.data_source = data_source
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        self._perm = torch.randperm(len(self.data_source), generator=g).tolist()

    def __iter__(self):
        # Reuse the same permutation each time (same order each epoch)
        return iter(self._perm)

    def __len__(self):
        return len(self.data_source)



def get_dataloader(ds, batch_size, seed=None):

    train_dl = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=FixedPermutationSampler(ds, seed=seed),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return train_dl

