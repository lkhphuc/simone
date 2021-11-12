import elegy as eg
import jax
import flax.linen as nn
import jax.numpy as jnp
import einops as E
import distrax as dist
from positional_encodings import PositionalEncoding3D


class _ConvBlock(nn.Module):
    conv_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.conv_dim, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.Conv(self.conv_dim, kernel_size=(3,3), strides=(2,2))(jax.nn.relu(x))
        x = nn.Conv(self.conv_dim, kernel_size=(3,3), strides=(2,2))(jax.nn.relu(x))

class TransformerEncoder():
    def __call__(self, x):
        pass

class Encoder(nn.Module):
    num_slots: int =16
    z_dim:int = 32
    conv_dim:int=128

    @nn.compact
    def forward(self,x):
        b, t,h,w,c = x.shape
        x = E.rearrange(x, 'b t h w c -> (b t) h w c')
        z = _ConvBlock(x) # b t 8 8 128
        del(x)

        z = E.rearrange(z, '(b t) h w -> b t c h w', t=t)
        z = self.pos(z) # TODO absolute positional embedding

        z = E.rearrange(z, 'b t h w c -> b (t h w) c')
        z = TransformerEncoder()(z)

        # Sum Pooling to get K slots
        z = E.rearrange(z, 'b (t h w) c -> b t h w c', t=t, h=8, w=8)
        z = E.reduce(z, 'b t (h h2) (w w2) c -> b t h w c', 'sum', h2=2, w2=2)

        z = E.rearrange(z, 'b t i j c -> b (t i j) c', t=t)
        z = TransformerEncoder()(z)

        z = E.rearrange(z, 'b (t k) c -> b k t c', t=t, k=self.num_slots)
        object_params = nn.Dense(self.z_dim*2)(E.reduce(z, 'b k t c -> b k c', 'mean'))
        object_posterior = dist.Normal(object_params[:,:,:self.z_dim], object_params[:,:,self.z_dim:].exp())
        frame_params = nn.Dense(self.z_dim*2)(E.reduce(z, 'b k t c -> b t c', 'mean'))
        frame_posterior = dist.Normal(frame_params[:,:,:self.z_dim], frame_params[:,:,self.z_dim:].exp())
        return object_posterior, frame_posterior


class Decoder(nn.Module):
    "Pixel-wise MLP decoder with 1x1 Conv"
    conv_dim:int = 128
    @nn.compact
    def __call__(self, z):
        z = nn.Conv(self.conv_dim, kernel_size=(1,1), strides=(1,1))(z)
        z = nn.Conv(self.conv_dim, kernel_size=(1,1), strides=(1,1))(jax.nn.relu(z))
        z = nn.Conv(self.conv_dim, kernel_size=(1,1), strides=(1,1))(jax.nn.relu(z))
        z = nn.Conv(self.conv_dim, kernel_size=(1,1), strides=(1,1))(jax.nn.relu(z))
        z = nn.Conv(self.conv_dim, kernel_size=(1,1), strides=(1,1))(jax.nn.relu(z))


class SIMONe(nn.Module):
    k = 16
    h = 64
    w = 64
    conv_dim=128

    @nn.nowrap
    def _coord_map(self):
        # a pair of coordinate channels one for each spatial dimension – ranging from -1 to 1.
        xs = jnp.linspace(-1, 1, self.h)
        ys = jnp.linspace(-1, 1, self.w)
        xb, yb = jnp.meshgrid(xs, ys, indexing='ij')
        return E.rearrange([xb,yb], 'h w -> h w c')

    @nn.compact 
    def forward(self, x):
        b, t, h, w, c = x.shape
        coord_map = E.repeat(self._coord_map(), 'h w c -> b k t h w c', b=b, k=self.k, t=t)
        timestep = E.repeat(jnp.arange(0, t), 't -> b k t h w 1', b=b, k=self.k, h=h, w=w)

        object_posterior, frame_posterior = Encoder(z_dim=32)(x)  # # Inference network

        object_latents = object_posterior.rsample([t, h, w])
        object_latents = E.rearrange(object_latents, 't h w b k c -> b k t h w c')
        frame_latents = frame_posterior.rsample([self.k, h, w])
        frame_latents = E.rearrange(frame_latents, 'k h w b t c -> b k t h w c')
        latents = jnp.concatenate([object_latents, frame_latents, coord_map, timestep], axis=-1)

        # TODO strided slice of inputs
        rec_x_masks = Decoder(self.conv_dim)(E.rearrange(latents, 'b k t h w c -> (b k t) c h w'))

        rec_x_masks = E.rearrange(rec_x_masks, '(b k t) c h w -> b k t c h w', b=b, k=self.k, t=t)
        rec_x, masks = rec_x_masks[:,:,:,:3,:,:], rec_x_masks[:,:,:,3:,:,:]
        masks = jax.nn.softmax(masks, axis=1) # softmax along the K object dims

        return rec_x, masks, object_posterior, frame_posterior


class SIMONeModel(eg.Model):
    module: eg.FlaxModule
    optimizer: eg.Optimizer
    alpha = 1
    beta_o = 0.5
    beta_f = 0.5

    def init_step(self, key: jnp.ndarray, inputs):
        self.module = self.module.init(key, inputs)
        self.optimizer = self.optimizer.init(self.parameters())
        return self

    def test_step(self, x):
        rec_x, masks, object_posterior, frame_posterior = self.module(x)
        rec_x_full = jnp.sum(rec_x*masks, axis=1) # Full frame equals mixture over slots
        del(rec_x, masks)

        p_x = dist.Normal(jax.nn.sigmoid(rec_x_full), jnp.ones_like(rec_x_full)*0.08)
        nll = -p_x.log_prob(x)
        nll = E.reduce(nll, 'b t c h w -> b t h w', 'sum').mean() # Mean over b+t+h+w

        object_prior = dist.Normal(0., 1.)
        object_kl = object_posterior.kl_divergence(object_prior)
        object_kl = E.reduce(object_kl, 'b k c -> b k', 'sum').mean() # Mean over b+k

        frame_prior = dist.Normal(0.0, 1.0)
        frame_kl = frame_posterior.kl_divergence(frame_prior)
        frame_kl = E.reduce(frame_kl, 'b t c -> b t', 'sum').mean() # Mean over b+t

        loss = self.alpha*nll + self.beta_o*object_kl + self.beta_f*frame_kl
        logs = {
            "loss/train": loss,
            "loss/nll": self.alpha*nll,
            "loss/latent_o": self.beta_o*object_kl,
            "loss/latent_f": self.beta_f*frame_kl,
        }

        return loss, logs, self



def _test():
    model = SIMONe()
    inputs = jnp.ones([1,10,3,64,64])
    out = model(inputs)
    print(out.shape)


if __name__ == "__main__":
    _test()
