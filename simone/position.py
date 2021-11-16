import jax
import jax.numpy as jnp
import einops as E


def embed_pos_3d(tensor: jnp.ndarray):
    assert tensor.ndim == 5, ("3D positional embedding only works for 5D Tensor")
    b, x, y, z, c = tensor.shape
    channels = int(jnp.ceil(c/6)*2)
    if channels % 2:
        channels += 1
    inv_freq = 1. / (10000 ** (jnp.arange(0, channels, 2) / channels))

    x_inp = jnp.einsum("i,j->ij", jnp.arange(x), inv_freq)
    y_inp = jnp.einsum("i,j->ij", jnp.arange(y), inv_freq)
    z_inp = jnp.einsum("i,j->ij", jnp.arange(z), inv_freq)

    emb_x = jnp.concatenate([jnp.sin(x_inp), jnp.cos(x_inp)], axis=-1)
    emb_y = jnp.concatenate([jnp.sin(y_inp), jnp.cos(y_inp)], axis=-1)
    emb_z = jnp.concatenate([jnp.sin(z_inp), jnp.cos(z_inp)], axis=-1)

    emb_x = E.repeat(emb_x, "x c -> b x y z c", b=b, y=y, z=z)
    emb_y = E.repeat(emb_y, "y c -> b x y z c", b=b, x=x, z=z)
    emb_z = E.repeat(emb_z, "z c -> b x y z c", b=b, x=x, y=y)

    return jnp.concatenate([emb_x, emb_y, emb_z], axis=-1)[...,:c]



def _test():
    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, [4, 32,32,3])


if __name__ == "__main__":
    _test()
