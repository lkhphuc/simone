from datetime import datetime
import typing as tp

import typer
import distrax as dist
import einops as E
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax.training.train_state import TrainState
from clu import metric_writers
import optax
import datasets
import elegy as eg

from model.monet import MONet


class MONetModel(eg.Model):
    module: eg.FlaxModule
    optimizer: eg.Optimizer

    def init_step(self, key: jnp.ndarray, inputs: tp.Any):
        self.next_key = eg.KeySeq(key)

        self.module.init_rngs = ("params", "qz")
        self.module.rngs = ("qz",)
        self.module = self.module.init(key, inputs)

        self.optimizer = self.optimizer.init(self.parameters())
        return self

    def test_step(self, inputs: tp.Any, labels: tp.Mapping[str, tp.Any]):
        q_z, imgs, masks = self.module(inputs)
        __import__('ipdb').set_trace()
        loss = self.compute_loss(q_z, imgs, masks)
        return loss, dict(loss=loss), self


    @staticmethod
    def compute_loss(q_z, images, masks_logit, beta=0.5, gamma=0.5):
      images_input, images_rec = images
      masks_input, masks_rec = masks_logit

      scale = jnp.ones_like(images_rec) * 0.11
      scale = scale.at[:,0].set(0.09)  # different scale for 1st slot
      p_x = dist.MultivariateNormalDiag(images_rec, scale)
      nll = -p_x.log_prob(images_input) * jnp.exp(masks_input.squeeze()) # Weighted by the masks
      nll = E.reduce(nll, 'b s h w -> b', 'sum').mean()

      p_z = dist.MultivariateNormalDiag(jnp.zeros_like(q_z.loc), jnp.ones_like(q_z.scale_diag))
      latent_kl = q_z.kl_divergence(p_z).mean()

      q_c = dist.Categorical(logits=E.rearrange(masks_input,     'b s h w c -> b h w c s'))
      p_c = dist.Categorical(logits=E.rearrange(masks_rec, 'b s c h w -> b h w c s'))
      mask_loss = q_c.kl_divergence(p_c).mean()

      return nll + beta * latent_kl + gamma * mask_loss


def main(
    eager:bool = False
    ):
    ds = datasets.load_dataset("data/ocrb.py")
    ds.set_format("numpy") # type:ignore
    train_dl = ds['train'].to_tf_dataset(columns=["video"], shuffle=True, batch_size=64).as_numpy_iterator()
    train_ds = jnp.array(ds['train'][:128]["video"], dtype=jnp.float32) / 255.0
    # val_ds = ds['validation'].to_tf_dataset(columns=["video"], shuffle=True, batch_size=64, prefetch=1)

    # X_train = jnp.load("data/datasets/vor/vor_train.npy")
    # X_train = X_train[:128,0].astype(jnp.float32) / 255.0
    # sample_imgs = next(iter(train_dl))[:5,0]

    model = MONetModel(
        module=MONet(num_slot=6),
        optimizer=optax.adagrad(1e-4),
        eager=eager,
    )

    model.fit(
        inputs=train_ds[:,0],
        batch_size=32,
        epochs=10,
        callbacks=[],
    )

    # _, images, masks_logit = eval_step(train_state, sample_imgs, rng)
    # imgs_input, imgs_rec = images
    # masks_input, masks_rec = masks_logit
    # x_cmb = jnp.sum(imgs_rec*masks_input, axis=1)
    # imgs = E.rearrange([imgs_input[:,0], x_cmb], ' s b h w c -> 1 (s h) (b w) c')
    # writer.write_images(epoch, {"Reconcstruction": imgs})


if __name__ == "__main__":
    typer.run(main)
