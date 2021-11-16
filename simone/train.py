from datetime import datetime

import einops as E
import elegy as eg
import jax
import jax.numpy as jnp
import optax
import typer
from tensorboardX import SummaryWriter

from simone import SIMONe, SIMONeModel
from dataloader import build_dataloader


class CustomLogger(eg.callbacks.TensorBoard):
    model: eg.Model
    train_writer: SummaryWriter
    def __init__(self, imgs, **kwargs):
        super().__init__(**kwargs)
        self.imgs = imgs

    def on_epoch_end(self, epoch, logs=None):
        # Log the same reconstruction image during training
        model = self.model.local()
        rec_x, masks, object_params, frame_params = model.predict(self.imgs)
        x = rec_x * masks
        x_cmb = jnp.sum(x, axis=1)
        plot_imgs = jnp.concatenate([self.imgs[:,None], x_cmb[:,None], x], axis=1)
        plot_imgs = E.rearrange(plot_imgs, " b k t h w c -> b (k h) (t w) c")
        if epoch % self.update_freq == 0:
            for i in range(4):
                self.train_writer.add_image(
                    f"Reconstruction/{i}", plot_imgs[i], epoch, dataformats="HWC"
                )


def main(
    lr: float = 1e-4,
    epochs: int = 100,
    nstep: int = None,
    eager: bool = False,
    profiling: bool = False,
):
    logdir = f"runs/{datetime.now()}"

    if profiling:
        jax.profiler.start_server(9999)
        jax.profiler.start_trace(logdir)

    train_dl, val_dl = build_dataloader(
        64, num_workers=24, dataset_class="vor", path="data/datasets", channel_last=True
    )

    sample_inp = next(iter(train_dl)).detach().numpy()[:8]

    model = SIMONeModel(
        module=SIMONe(),
        optimizer=optax.adamw(lr),
        eager=eager,
    )

    model.summary(sample_inp)
    model = model.distributed()

    history = model.fit(
        inputs=train_dl,
        validation_data=val_dl,
        # batch_size=32,
        # steps_per_epoch=nstep,
        epochs=epochs,
        callbacks=[
            eg.callbacks.TensorBoard(logdir=logdir),
            CustomLogger(sample_inp, logdir=logdir),
        ],
    )


    if profiling:
        jax.profiler.stop_trace()
        jax.profiler.save_device_memory_profile("memory.prof")


if __name__ == "__main__":
    typer.run(main)
