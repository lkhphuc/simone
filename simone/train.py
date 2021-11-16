from datetime import datetime

import einops as E
import elegy as eg
import jax
import jax.numpy as jnp
import optax
import typer
from tensorboardX import SummaryWriter

from data.dataloader import build_dataloader
from simone import SIMONe, SIMONeModel


class CustomLogger(eg.callbacks.TensorBoard):
    model: eg.Model
    train_writer: SummaryWriter
    def __init__(self, imgs, **kwargs):
        super().__init__(**kwargs)
        self.imgs = imgs

    def on_epoch_end(self, epoch, logs=None):
        # Log the same reconstruction image during training
        model = self.model.local()
        (imgs_inp, imgs_rec), (masks_inp, _), (_, _) = model.predict(self.imgs)
        x_cmb = jnp.sum(imgs_rec * masks_inp, axis=1)
        imgs = E.rearrange([imgs_inp[:, 0], x_cmb], " s b h w c -> (s h) (b w) c")
        if epoch % self.update_freq == 0:
            self.train_writer.add_image(
                "Reconstruction", imgs, epoch, dataformats="HWC"
            )


def main(
    lr: float = 1e-4,
    epochs: int = 100,
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
    jax.profiler.save_device_memory_profile("memory-model-start.prof")

    history = model.fit(
        inputs=train_dl,
        validation_data=val_dl,
        # batch_size=32,
        # steps_per_epoch=200,
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
