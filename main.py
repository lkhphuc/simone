from datetime import datetime

import einops as E
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax.training.train_state import TrainState
from clu import metric_writers
import optax
import datasets

from model.monet import MONet, loss_fn


# @jax.jit
def train_step(state, batch, rng):
    @jax.value_and_grad
    def loss_and_grad(params):
        q_z, images, masks_logit = state.apply_fn(params, batch, rngs={"qz": rng})
        loss = loss_fn(q_z, images, masks_logit)
        return loss

    loss, grads = loss_and_grad(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# @jax.jit
def eval_step(state, batch, rng):
    q_z, images, masks_logit = state.apply_fn(state.params, batch, rngs={"qz": rng})
    loss = loss_fn(q_z, images, masks_logit)
    return loss, images, masks_logit


def main():
    ds = datasets.load_dataset("data/ocrb.py")
    ds.set_format("numpy") # type:ignore
    train_dl = ds['train'].to_tf_dataset(columns=["video"], shuffle=True, batch_size=64).as_numpy_iterator()

    # val_ds = ds['validation'].to_tf_dataset(columns=["video"], shuffle=True, batch_size=64, prefetch=1)
    sample_imgs = next(iter(train_dl))[:5,0]

    rng = jax.random.PRNGKey(0)
    param_rng, qz_rng, rng = jax.random.split(rng, 3)

    model = MONet(num_slot=6)
    params = model.init({"params": param_rng, "qz": qz_rng}, sample_imgs)
    optimizer = optax.adagrad(1e-4)

    train_state = TrainState.create( #type:ignore
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    logdir = f"runs/{datetime.now()}"
    writer = metric_writers.create_default_writer(logdir)

    for epoch in range(10):
        qz_rng, rng = jax.random.split(rng)
        for batch in train_dl:
            train_state, loss = train_step(train_state, batch[:,0], rng)
            writer.write_scalars(epoch, dict(loss=loss))

        _, images, masks_logit = eval_step(train_state, sample_imgs, rng)
        imgs_input, imgs_rec = images
        masks_input, masks_rec = masks_logit
        x_cmb = jnp.sum(imgs_rec*masks_input, axis=1)
        imgs = E.rearrange([imgs_input[:,0], x_cmb], ' s b h w c -> 1 (s h) (b w) c')
        writer.write_images(epoch, {"Reconcstruction": imgs})


if __name__ == "__main__":
    main()
