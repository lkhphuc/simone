from pathlib import Path
from torch.utils import data
import numpy as np
import einops as E
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, file: str, n_steps=-1, transform=None, T=0, path="./data/datasets/", channel_last=False):
        self.transform = transform
        imgs : np.ndarray = np.load(Path(path, file))
        imgs = imgs[:, :n_steps]
        if channel_last:
            imgs = E.rearrange(imgs, 'b t c h w -> b t h w c')
        if T and T < n_steps:
            imgs = np.concatenate(np.split(imgs, imgs.shape[1]//T, axis=1))
        self.imgs = imgs.astype(np.float32) / 255.0

    def __getitem__(self, index):
        x = self.imgs[index]
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.imgs)


def build_dataloader(batch_size, num_workers=1, n_steps=10, dataset_class='vmds', path='./data/datasets', T=0, channel_last=False):

    # tform = torchvision.transforms.Lambda(lambda n: n / 255.)
    tform = None

    train_loader = data.DataLoader(
            SyntheticDataset(f"{dataset_class}/{dataset_class}_train.npy", n_steps=n_steps, transform=tform, path=path, T=T, channel_last=channel_last),
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, pin_memory=True)

    val_loader = data.DataLoader(
            SyntheticDataset(f"{dataset_class}/{dataset_class}_val.npy", n_steps=n_steps, transform=tform, path=path, T=T, channel_last=channel_last),
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def build_testloader(batch_size, num_workers=1, n_steps=10, dataset_class='vmds', path='./data/datasets', channel_last=False):

    # tform = torchvision.transforms.Lambda(lambda n: n / 255.)
    tform = None

    test_loader = data.DataLoader(
            SyntheticDataset(f"{dataset_class}/{dataset_class}_test.npy", n_steps=n_steps, transform=tform, path=path, channel_last=channel_last),
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, pin_memory=True)

    return test_loader


def _test():
    train_dl, val_dl = build_dataloader(16, dataset_class="vor")
    __import__('ipdb').set_trace()

if __name__ == "__main__":
    _test()
