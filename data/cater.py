from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms.functional as T
import skvideo.io
import ffmpeg


class CaterDataset(data.Dataset):
    def __init__(self, path:str, size=64, nframes=16, subset="train", transform=None):
        self.path = path
        with open(path+"/lists/localize/"+subset+".txt") as file:
            videos = file.readlines()
        self.videos = [video.split(" ")[0] for video in videos]
        self.size = size
        self.nframes = nframes
        self.tfms = transform

    def __getitem__(self, idx):
        vidgen = skvideo.io.vreader(self.path+"/videos/"+str(self.videos[idx]))
        frames = []
        for iframe, frame, in enumerate(vidgen):
            frame = np.array(frame[:,40:280,:]).transpose([2,0,1])
            frame = torch.from_numpy(frame)
            frame = T.resize(frame, (self.size, self.size))
            # frame = jax.image.resize(frame, (self.size, self.size, 3), 'bilinear', antialias=True)
            frames.append(frame)
            if iframe == self.nframes-1:
                break

        # frames = np.stack(frames)
        frames = torch.stack(frames).permute([0,2,3,1])
        # print(str(self.videos[idx]), frames.shape)

        if self.tfms:
            frames = self.tfms(frames)

        return frames

    def __len__(self):
        return len(self.videos)


def build_dataloader(batch_size, num_workers=1, path='./data/datasets/CATER/', size=64, nframes=16, channel_last=False, transform=None):

    # tform = torchvision.transforms.Lambda(lambda n: n / 255.)
    tform = lambda n: n/255.

    train_loader = data.DataLoader(
            CaterDataset(path, size=size, nframes=nframes, subset='train', transform=tform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers, pin_memory=True)

    val_loader = data.DataLoader(
            CaterDataset(path, size, nframes, 'val', transform=tform),
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def _test():
    ds = CaterDataset("data/datasets/CATER/")
    vid1 = ds[0]
    train_ds, val_ds = build_dataloader(10, channel_last=True)
    __import__('ipdb').set_trace()
    batch = next(iter(train_ds))

if __name__ == "__main__":
    _test()