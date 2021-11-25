import distrax as dist
import einops as E
import elegy as eg
import flax.linen as nn
import jax
import jax.numpy as jnp
from position import PositionalEmbedding3D


class ConvBlock(nn.Module):
    conv_dim: int = 128
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.conv_dim, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.Conv(self.conv_dim, kernel_size=(3, 3), strides=(2, 2))(jax.nn.relu(x))
        x = nn.Conv(self.conv_dim, kernel_size=(3, 3), strides=(2, 2))(jax.nn.relu(x))
        return x


class TransformerBlock(nn.Module):
    nhead: int = 4
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, x):
        channel_dim = x.shape[-1]
        x_res = nn.SelfAttention(self.nhead)(x)
        x = nn.LayerNorm()(x + x_res)

        x_res = nn.Dense(self.hidden_dim)(x)
        x_res = nn.Dense(channel_dim)(jax.nn.relu(x_res))

        x = nn.LayerNorm()(x + x_res)
        return x


class TransformerEncoder(nn.Module):
    nlayer: int = 4

    @nn.compact
    def __call__(self, x):
        for _ in range(self.nlayer):
            x = TransformerBlock()(x)
        return x


class Encoder(nn.Module):
    num_slots: int = 16
    z_dim: int = 32

    @nn.compact
    def __call__(self, x):
        b, t, h, w, c = x.shape

        x = E.rearrange(x, "b t h w c -> (b t) h w c")
        z = ConvBlock()(x)  # b t 8 8 128
        z = E.rearrange(z, "(b t) h w c -> b t h w c", t=t)

        _, _, z_h, z_w, z_c = z.shape
        z = PositionalEmbedding3D(128)(z)
        z = E.rearrange(z, "b t h w c -> b (t h w) c")
        z = TransformerEncoder()(z)
        z = E.rearrange(z, "b (t h w) c -> b t h w c", t=t, h=z_h, w=z_w)

        # Sum Pooling then scale to get K slots
        z = E.reduce(z, "b t (i h) (j w) c -> b t i j c", "sum", h=2, w=2)
        z = z / 2  # 16 / (8*8) = 1/2

        z = E.rearrange(z, "b t i j c -> b (t i j) c", t=t)
        z = TransformerEncoder()(z)
        z = E.rearrange(z, "b (t k) c -> b k t c", t=t, k=self.num_slots) # NOTE: reverse K<->T

        obj_params = nn.Dense(1024)(E.reduce(z, "b k t c -> b k c", "mean"))
        obj_params = nn.Dense(self.z_dim*2)(jax.nn.relu(obj_params))
        frame_params = nn.Dense(1024)(E.reduce(z, "b k t c -> b t c", "mean"))
        frame_params = nn.Dense(self.z_dim*2)(jax.nn.relu(frame_params))
        return obj_params, frame_params


class Decoder(nn.Module):
    "Pixel-wise MLP decoder with 1x1 Conv"
    nchannels: int = 512

    @nn.compact
    def __call__(self, z):
        z = jax.nn.relu(nn.Conv(self.nchannels, kernel_size=(1, 1), strides=(1, 1))(z))
        z = jax.nn.relu(nn.Conv(self.nchannels, kernel_size=(1, 1), strides=(1, 1))(z))
        z = jax.nn.relu(nn.Conv(self.nchannels, kernel_size=(1, 1), strides=(1, 1))(z))
        z = jax.nn.relu(nn.Conv(self.nchannels, kernel_size=(1, 1), strides=(1, 1))(z))
        z = jax.nn.relu(nn.Conv(self.nchannels, kernel_size=(1, 1), strides=(1, 1))(z))
        z = jax.nn.relu(nn.Conv(self.nchannels, kernel_size=(1, 1), strides=(1, 1))(z))
        return nn.Conv(4, kernel_size=(1, 1), strides=(1, 1))(z)


class SIMONeModel(eg.Model):
    # module: eg.FlaxModule
    def __init__(self,
        conv_dim = 128,
        nlayer = 4,
        nhead = 4,
        alpha = 0.2,
        z_dim = 32,
        k = 16,  # Number of objects
        beta_o = 1e-8,
        beta_f = 1e-8,
        loss = None,
        metrics = None,
        optimizer=None,
        seed = 42,
        eager = False,
    ):

        self.conv_dim = conv_dim
        self.nlayer = nlayer
        self.nhead = nhead
        self.alpha = alpha
        self.z_dim = z_dim
        self.k = k
        self.beta_o = beta_o
        self.beta_f = beta_f
        self.encoder = eg.FlaxModule(Encoder(z_dim=z_dim))
        self.decoder = eg.FlaxModule(Decoder(conv_dim))
        super().__init__(
            module=None,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            seed=seed,
            eager=eager,
        )
        

    def init_step(self, key: jnp.ndarray, inputs):
        self.next_key = eg.KeySeq(key)
        self.b, self.t, self.h, self.w, self.c = inputs.shape
        self.encoder = self.encoder.init(key, inputs)
        self.decoder = self.decoder.init(key, jnp.ones([2, self.h, self.w, self.z_dim*2+3]))

        if self.optimizer:
            self.optimizer = self.optimizer.init(self.parameters())

        return self


    def _coord_map(self):
        # a pair of coordinate channels one for each spatial dimension – ranging from -1 to 1.
        xs = jnp.linspace(-1, 1, self.h)
        ys = jnp.linspace(-1, 1, self.w)
        xb, yb = jnp.meshgrid(xs, ys, indexing="ij")
        return E.rearrange([xb, yb], "c h w -> h w c")

    def encode(self, x):
        object_params, frame_params = self.encoder(x)

        object_posterior = dist.MultivariateNormalDiag(
            object_params[..., : self.z_dim], jnp.exp(object_params[..., self.z_dim :])
        )

        frame_posterior = dist.MultivariateNormalDiag(
            frame_params[:, :, : self.z_dim], jnp.exp(frame_params[:, :, self.z_dim :])
        )
        return object_posterior, frame_posterior


    def decode(self, object_posterior, frame_posterior, x):
        b, t, h, w, c = x.shape
        object_latents = object_posterior.sample(
            seed=self.next_key(), sample_shape=[t, h, w]
        )
        object_latents = E.rearrange(object_latents, "t h w b k c -> b k t h w c")

        frame_latents = frame_posterior.sample(
            seed=self.next_key(), sample_shape=[self.k, h, w]
        )
        frame_latents = E.rearrange(frame_latents, "k h w b t c -> b k t h w c")
        coord_map = E.repeat(
            self._coord_map(), "h w c -> b k t h w c", b=b, k=self.k, t=t
        )
        timestep = E.repeat(
            jnp.arange(0, self.t), "t -> b k t h w 1", b=b, k=self.k, h=h, w=w
        )

        latents = jnp.concatenate(
            [object_latents, frame_latents, coord_map, timestep],
            axis=-1
        )
        latents = E.rearrange(latents, "b k t h w c -> (b k t) h w c")

        # TODO strided slice of inputs
        rec_x_masks = self.decoder(latents)
        rec_x_masks = E.rearrange(
            rec_x_masks, "(b k t) h w c -> b k t h w c", b=b, k=self.k, t=t
        )
        rec_x = jax.nn.sigmoid(rec_x_masks[..., :3])
        # softmax along the K object dims
        masks = jax.nn.softmax(rec_x_masks[..., 3:], axis=1)
        return rec_x, masks

    def pred_step(self, inputs):
        object_posterior, frame_posterior = self.encode(inputs)
        rec_x, masks = self.decode(object_posterior, frame_posterior, inputs)
        return (rec_x, masks), self

    def test_step(self, x, labels):
        object_posterior, frame_posterior = self.encode(x)
        rec_x, masks = self.decode(object_posterior, frame_posterior, x)
        # Full frame equals mixture over slots
        rec_x_full = jnp.sum(rec_x * masks, axis=1)  # b t h w c

        p_x = dist.MultivariateNormalDiag(rec_x_full, jnp.ones_like(rec_x_full) * 0.08)
        nll = -p_x.log_prob(x).mean()  # Mean over b+t+h+w

        object_prior = dist.MultivariateNormalDiag(
            jnp.zeros_like(object_posterior.loc),
            jnp.ones_like(object_posterior.scale_diag),
        )
        object_kl = object_posterior.kl_divergence(object_prior).mean()  # Mean over b+k
        frame_prior = dist.MultivariateNormalDiag(
            jnp.zeros_like(frame_posterior.loc),
            jnp.ones_like(frame_posterior.scale_diag),
        )
        frame_kl = frame_posterior.kl_divergence(frame_prior).mean()  # Mean over b+t

        loss = self.alpha * nll + self.beta_o * object_kl + self.beta_f * frame_kl
        logs = {
                "loss/total": loss,
                "loss/nll": self.alpha * nll,
                "loss/latent_o": self.beta_o * object_kl,
                "loss/latent_f": self.beta_f * frame_kl,
                }

        return loss, logs, self


def _test():
    inputs = jnp.ones([2, 10, 64, 64, 3])

    # model = SIMONe()
    # out, params = model.init_with_output(
    #     {"params": jax.random.PRNGKey(42), "latent": jax.random.PRNGKey(12)}, inputs
    # )

    __import__('ipdb').set_trace()
    model = SIMONeModel(optimizer=None)
    model.init_on_batch(inputs)
    out = model.encode(inputs)
    rec_x, masks, = model.decode(*out, inputs)
    # model.summary(inputs)



if __name__ == "__main__":
    _test()
