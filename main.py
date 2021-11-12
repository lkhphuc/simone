from datetime import datetime

import einops as E
import elegy as eg
import jax
import jax.numpy as jnp
import optax
import typer

from data.dataloader import build_dataloader
from model.monet import MONet, MONetModel


class MONetCustomLogger(eg.callbacks.TensorBoard):
    def __init__(self, imgs, **kwargs):
        super().__init__(**kwargs)
        self.imgs = imgs

    def on_epoch_end(self, epoch, logs=None):
        # Log the same reconstruction image during training
        (imgs_inp, imgs_rec), (masks_inp, _), (_, _) = self.model.predict(
            self.imgs
        )  # type:ignore
        x_cmb = jnp.sum(imgs_rec * masks_inp, axis=1)
        imgs = E.rearrange([imgs_inp[:, 0], x_cmb], " s b h w c -> (s h) (b w) c")
        if epoch % self.update_freq == 0:
            self.train_writer.add_image(
                "Reconstruction", imgs, self.global_step, dataformats="HWC"
            )
        jax.profiler.save_device_memory_profile(f"memory-model-{epoch}.prof")


def main(
    num_slot: int = 6,
    lr: float = 1e-4,
    epochs: int = 1,
    eager: bool = False,
    profiling: bool = False,
):
    logdir = f"runs/{datetime.now()}"

    if profiling:
        jax.profiler.start_server(9999)
        jax.profiler.start_trace(logdir)

    train_dl, val_dl = build_dataloader(
        64, num_workers=4, dataset_class="vor", path="data/datasets", channel_last=True
    )

    sample_inp = next(iter(train_dl)).detach().numpy()[:5]

    model = MONetModel(
        module=MONet(num_slot=num_slot),
        optimizer=optax.adamw(lr),
        eager=eager,
    )

    model.summary(sample_inp)
    jax.profiler.save_device_memory_profile("memory-model-start.prof")

    history = model.fit(
        inputs=train_dl,
        steps_per_epoch=10,
        # batch_size=32,
        validation_data=val_dl,
        epochs=epochs,
        callbacks=[
            eg.callbacks.TensorBoard(logdir=logdir),
            MONetCustomLogger(sample_inp, logdir=logdir),
        ],
    )

    if profiling:
        jax.profiler.stop_trace()
        jax.profiler.save_device_memory_profile("memory.prof")


if __name__ == "__main__":
    typer.run(main)
