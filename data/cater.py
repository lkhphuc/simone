import numpy as np
import skvideo.io
import torch
import torchvision.transforms.functional as T
from torch.utils import data


class CaterDataset(data.Dataset):
    def __init__(self, path: str, size=64, nframes=16, subset="train", transform=None):
        self.path = path
        with open(path + "/lists/localize/" + subset + ".txt") as file:
            videos = file.readlines()
        self.videos = [video.split(" ")[0] for video in videos]
        self.size = size
        self.nframes = nframes
        self.tfms = transform

    def __getitem__(self, idx):
        vidgen = skvideo.io.vreader(self.path + "/videos/" + str(self.videos[idx]))
        frames = []
        for (
            iframe,
            frame,
        ) in enumerate(vidgen):
            frame = np.array(frame[:, 40:280, :]).transpose([2, 0, 1])
            frame = torch.from_numpy(frame)
            frame = T.resize(frame, (self.size, self.size))
            # frame = jax.image.resize(frame, (self.size, self.size, 3), 'bilinear', antialias=True)
            frames.append(frame)
            if iframe == self.nframes - 1:
                break

        # frames = np.stack(frames)
        frames = torch.stack(frames).permute([0, 2, 3, 1])
        # print(str(self.videos[idx]), frames.shape)

        if self.tfms:
            frames = self.tfms(frames)

        return frames

    def __len__(self):
        return len(self.videos)


def build_dataloader(
    subset,
    batch_size,
    num_workers=1,
    path="./data/datasets/CATER/",
    size=64,
    nframes=16,
    shuffle=False,
    transform=None,
    prefetch_factor=2,
):

    # tform = torchvision.transforms.Lambda(lambda n: n / 255.)
    if not transform:
        transform = lambda n: n / 255.0

    loader = data.DataLoader(
        CaterDataset(
            path, size=size, nframes=nframes, subset=subset, transform=transform
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
    )

    return loader


def _dev():
    ds = CaterDataset("data/datasets/CATER/")
    vid1 = ds[0]
    train_dl = build_dataloader("train", 10)
    val_dl = build_dataloader("val", 10)
    batch = next(iter(train_dl))
    __import__("ipdb").set_trace()


if __name__ == "__main__":
    _dev()
