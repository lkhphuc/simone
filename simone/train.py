from datetime import datetime

import einops as E
import elegy as eg
import jax
import jax.numpy as jnp
import optax
import typer
from tensorboardX import SummaryWriter

from model_new import SIMONeModel, Encoder, Decoder
# from dataloader import build_dataloader
from data.cater import build_dataloader


class CustomLogger(eg.callbacks.TensorBoard):
    model: eg.Model
    train_writer: SummaryWriter
    def __init__(self, imgs, **kwargs):
        super().__init__(**kwargs)
        self.imgs = imgs

    def on_epoch_end(self, epoch, logs={}):
        # Log the same reconstruction image during training
        model = self.model.local()
        rec_x, masks = model.predict(self.imgs)

        def norm_ip(img, low, high):
            img = jnp.clip(img, low, high)
            img = (img - low) / max(high - low, 1e-5)
            return img

        x_cmb = jnp.sum(rec_x * masks, axis=1)
        plot_imgs = jnp.concatenate([self.imgs[:,None], x_cmb[:,None], rec_x], axis=1)
        plot_imgs = E.rearrange(plot_imgs, " b k t h w c -> b (k h) (t w) c")
        # plot_imgs = norm_ip(plot_imgs, 0, 1)
        if epoch % self.update_freq == 0:
            for i in range(4):
                self.train_writer.add_image(
                    f"Reconstruction/{i}", plot_imgs[i], epoch, dataformats="HWC"
                )


def main(
    lr: float = 20e-5,
    bs: int = 6,
    epochs: int = 1000,
    nstep: int = None,
    eager: bool = False,
    profiling: bool = False,
    distributed: bool = True,
):
    logdir = f"runs/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ndevice = 1
    if distributed:
        ndevice = jax.device_count()

    if profiling:
        eager = False
        jax.profiler.start_server(9999)
        jax.profiler.start_trace(logdir)


    # train_dl, val_dl = build_dataloader(
    #     ndevice*bs, num_workers=ndevice*2, dataset_class="vor", path="data/datasets", channel_last=True
    # )
    train_dl, val_dl = build_dataloader(
        ndevice*bs, num_workers=ndevice*4, path="data/datasets/CATER/", channel_last=True
    )

    sample_inp = next(iter(train_dl)).detach().numpy()[:ndevice]

    model = SIMONeModel(
        # module=SIMONe(),
        optimizer=optax.adamw(lr),
        eager=eager,
    )

    # model.summary(sample_inp)
    if distributed:
        model = model.distributed()

    history = model.fit(
        inputs=train_dl,
        validation_data=val_dl,
        batch_size=bs,
        steps_per_epoch=nstep,
        # validation_steps=10,
        epochs=epochs,
        callbacks=[
            eg.callbacks.TensorBoard(logdir=logdir),
            eg.callbacks.ModelCheckpoint(path=logdir, monitor="val_loss/total", save_best_only=True),
            CustomLogger(sample_inp, logdir=logdir),
        ],
    )


    if profiling:
        jax.profiler.stop_trace()
        jax.profiler.save_device_memory_profile("memory.prof")


if __name__ == "__main__":
    typer.run(main)
