import os
from torch.utils import data
import torchvision
import numpy as np
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, mode='train', n_steps=10, dataset_class='vmds', transform=None, path="", T=0):
        assert dataset_class in ['vmds', 'vor', 'spmot']
        self.transform = transform
        path = os.path.join(path, dataset_class, '{}_{}.npy'.format(dataset_class, mode))
        imgs : np.ndarray = np.load(path)
        imgs = imgs[:, :n_steps]
        if T and T < n_steps:
            imgs = np.concatenate(np.split(imgs, imgs.shape[1]//T, axis=1))
        self.imgs = imgs.squeeze().astype(np.float32) # remove temporal dimension if == 1

    def __getitem__(self, index):
        x = self.imgs[index]
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.imgs)


def build_dataloader(batch_size, num_workers=1, n_steps=10, dataset_class='vmds', path='./data/data', T=0):

    tform = torchvision.transforms.Lambda(lambda n: n / 255.)

    train_loader = data.DataLoader(
            SyntheticDataset(mode='train', n_steps=n_steps, transform=tform, dataset_class=dataset_class, path=path, T=T),
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, pin_memory=True)

    val_loader = data.DataLoader(
            SyntheticDataset(mode='val', n_steps=n_steps, transform=tform, dataset_class=dataset_class, path=path, T=T),
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def build_testloader(batch_size, num_workers=1, n_steps=10, dataset_class='vmds', path='./data/data'):

    tform = torchvision.transforms.Lambda(lambda n: n / 255.)

    test_loader = data.DataLoader(
            SyntheticDataset(mode='test', n_steps=n_steps, transform=tform, dataset_class=dataset_class, path=path),
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, pin_memory=True)

    return test_loader
