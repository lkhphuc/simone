import distrax as dist
import einops as E
import elegy as eg
import flax.linen as nn
import jax
import jax.numpy as jnp
from position import embed_pos_3d


class ConvBlock(nn.Module):
    conv_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.conv_dim, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.Conv(self.conv_dim, kernel_size=(3, 3), strides=(2, 2))(jax.nn.relu(x))
        x = nn.Conv(self.conv_dim, kernel_size=(3, 3), strides=(2, 2))(jax.nn.relu(x))
        return x


class TransformerBlock(nn.Module):
    nhead: int
    ndim: int

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        x_res = nn.SelfAttention(self.nhead)(x)
        x = nn.LayerNorm()(x + x_res)

        x_res = nn.Dense(self.ndim)(x)
        x_res = nn.Dense(dim)(jax.nn.relu(x_res))

        x = nn.LayerNorm()(x + x_res)
        return x


class TransformerEncoder(nn.Module):
    nlayer: int = 2
    nhead: int = 2
    ndim: int = 64

    @nn.compact
    def __call__(self, x):
        for _ in range(self.nlayer):
            x = TransformerBlock(self.nhead, self.ndim)(x)
        return x


class Encoder(nn.Module):
    num_slots: int = 16
    z_dim: int = 32
    conv_dim: int = 128
    nlayer: int = 4
    nhead: int = 5
    ndim: int = 1024

    @nn.compact
    def __call__(self, x):
        b, t, h, w, c = x.shape
        x = E.rearrange(x, "b t h w c -> (b t) h w c")
        z = ConvBlock(self.conv_dim)(x)  # b t 8 8 128

        z = E.rearrange(z, "(b t) h w c -> b t h w c", t=t)
        z = embed_pos_3d(z) + z

        z = E.rearrange(z, "b t h w c -> b (t h w) c")
        z = TransformerEncoder()(z)

        # Sum Pooling to get K slots
        z = E.rearrange(z, "b (t h w) c -> b t h w c", t=t, h=8, w=8)
        z = E.reduce(z, "b t (h h2) (w w2) c -> b t h w c", "sum", h2=2, w2=2)

        z = E.rearrange(z, "b t i j c -> b (t i j) c", t=t)
        z = TransformerEncoder()(z)

        z = E.rearrange(z, "b (t k) c -> b k t c", t=t, k=self.num_slots)
        object_params = nn.Dense(self.z_dim * 2)(
            E.reduce(z, "b k t c -> b k c", "mean")
        )
        object_posterior = dist.MultivariateNormalDiag(
            object_params[..., : self.z_dim], jnp.exp(object_params[..., self.z_dim :])
        )
        frame_params = nn.Dense(self.z_dim * 2)(E.reduce(z, "b k t c -> b t c", "mean"))
        frame_posterior = dist.MultivariateNormalDiag(
            frame_params[:, :, : self.z_dim], jnp.exp(frame_params[:, :, self.z_dim :])
        )
        return object_posterior, frame_posterior


class Decoder(nn.Module):
    "Pixel-wise MLP decoder with 1x1 Conv"
    conv_dim: int = 128

    @nn.compact
    def __call__(self, z):
        z = jax.nn.relu(nn.Conv(self.conv_dim, kernel_size=(1, 1), strides=(1, 1))(z))
        z = jax.nn.relu(nn.Conv(self.conv_dim, kernel_size=(1, 1), strides=(1, 1))(z))
        z = jax.nn.relu(nn.Conv(self.conv_dim, kernel_size=(1, 1), strides=(1, 1))(z))
        z = jax.nn.relu(nn.Conv(self.conv_dim, kernel_size=(1, 1), strides=(1, 1))(z))
        z = nn.Conv(4, kernel_size=(1, 1), strides=(1, 1))(z)
        return z


class SIMONe(nn.Module):
    k = 16
    h = 64
    w = 64
    conv_dim = 128
    nlayer = 4
    nhead = 4
    z_dim = 32

    @nn.nowrap
    def _coord_map(self):
        # a pair of coordinate channels one for each spatial dimension – ranging from -1 to 1.
        xs = jnp.linspace(-1, 1, self.h)
        ys = jnp.linspace(-1, 1, self.w)
        xb, yb = jnp.meshgrid(xs, ys, indexing="ij")
        return E.rearrange([xb, yb], "c h w -> h w c")

    @nn.compact
    def __call__(self, x):
        b, t, h, w, c = x.shape
        coord_map = E.repeat(
            self._coord_map(), "h w c -> b k t h w c", b=b, k=self.k, t=t
        )
        timestep = E.repeat(
            jnp.arange(0, t), "t -> b k t h w 1", b=b, k=self.k, h=h, w=w
        )

        object_posterior, frame_posterior = Encoder(z_dim=32, conv_dim=self.conv_dim)(
            x
        )  # # Inference network

        object_latents = object_posterior.sample(
            seed=self.make_rng("latent"), sample_shape=[t, h, w]
        )
        object_latents = E.rearrange(object_latents, "t h w b k c -> b k t h w c")
        frame_latents = frame_posterior.sample(
            seed=self.make_rng("latent"), sample_shape=[self.k, h, w]
        )
        frame_latents = E.rearrange(frame_latents, "k h w b t c -> b k t h w c")
        latents = jnp.concatenate(
            [object_latents, frame_latents, coord_map, timestep], axis=-1
        )

        # TODO strided slice of inputs
        rec_x_masks = Decoder(self.conv_dim)(
            E.rearrange(latents, "b k t h w c -> (b k t) h w c")
        )

        rec_x_masks = E.rearrange(
            rec_x_masks, "(b k t) h w c -> b k t h w c", b=b, k=self.k, t=t
        )
        rec_x, masks = rec_x_masks[..., :3], rec_x_masks[..., 3:]
        masks = jax.nn.softmax(masks, axis=1)  # softmax along the K object dims

        return rec_x, masks, object_posterior, frame_posterior


class SIMONeModel(eg.Model):
    module: eg.FlaxModule
    optimizer: eg.Optimizer
    alpha = 1
    beta_o = 0.5
    beta_f = 0.5

    # def init_step(self, key: jnp.ndarray, inputs):
    #     self.module = self.module.init(key, inputs)
    #     self.optimizer = self.optimizer.init(self.parameters())
    #     return self

    def test_step(self, x):
        rec_x, masks, object_posterior, frame_posterior = self.module(x)
        # Full frame equals mixture over slots
        rec_x_full = jnp.sum(rec_x * masks, axis=1)

        p_x = dist.Normal(jax.nn.sigmoid(rec_x_full), jnp.ones_like(rec_x_full) * 0.08)
        nll = -p_x.log_prob(x)
        nll = E.reduce(nll, "b t c h w -> b t h w", "sum").mean()  # Mean over b+t+h+w

        object_prior = dist.Normal(0.0, 1.0)
        object_kl = object_posterior.kl_divergence(object_prior)
        object_kl = E.reduce(object_kl, "b k c -> b k", "sum").mean()  # Mean over b+k

        frame_prior = dist.Normal(0.0, 1.0)
        frame_kl = frame_posterior.kl_divergence(frame_prior)
        frame_kl = E.reduce(frame_kl, "b t c -> b t", "sum").mean()  # Mean over b+t

        loss = self.alpha * nll + self.beta_o * object_kl + self.beta_f * frame_kl
        logs = {
            "loss/train": loss,
            "loss/nll": self.alpha * nll,
            "loss/latent_o": self.beta_o * object_kl,
            "loss/latent_f": self.beta_f * frame_kl,
        }

        return loss, logs, self


def _test():
    model = SIMONe()
    inputs = jnp.ones([2, 10, 64, 64, 3])
    out, params = model.init_with_output(
        {"params": jax.random.PRNGKey(42), "latent": jax.random.PRNGKey(12)}, inputs
    )
    print(out.shape)


if __name__ == "__main__":
    _test()
