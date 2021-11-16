import einops as E
import flax.linen as nn
import jax
import jax.numpy as jnp


class PositionalEmbedding3D(nn.Module):
    channels: int

    def setup(self):
        self.inv_freq = 1.0 / (
            10000 ** (jnp.arange(0, self.channels, 2) / self.channels)
        )

    def __call__(self, tensor):
        assert tensor.ndim == 5, "3D positional embedding only works for 5D Tensor"
        assert (
            tensor.shape[-1] <= self.channels
        ), "Channel dimension must be smaller than the maximum channels"
        b, x, y, z, c = tensor.shape  # Explicitly passed in c_dim for jax

        x_inp = jnp.einsum("i,j->ij", jnp.arange(x), self.inv_freq)
        y_inp = jnp.einsum("i,j->ij", jnp.arange(y), self.inv_freq)
        z_inp = jnp.einsum("i,j->ij", jnp.arange(z), self.inv_freq)

        emb_x = jnp.concatenate([jnp.sin(x_inp), jnp.cos(x_inp)], axis=-1)
        emb_y = jnp.concatenate([jnp.sin(y_inp), jnp.cos(y_inp)], axis=-1)
        emb_z = jnp.concatenate([jnp.sin(z_inp), jnp.cos(z_inp)], axis=-1)

        emb_x = E.repeat(emb_x, "x c -> b x y z c", b=b, y=y, z=z)
        emb_y = E.repeat(emb_y, "y c -> b x y z c", b=b, x=x, z=z)
        emb_z = E.repeat(emb_z, "z c -> b x y z c", b=b, x=x, y=y)

        return jnp.concatenate([emb_x, emb_y, emb_z], axis=-1)[..., :c] + tensor


def _test():
    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, [4, 32, 32, 3])


if __name__ == "__main__":
    _test()
