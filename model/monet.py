import jax
import distrax as dist
import flax.linen as nn
import jax.numpy as jnp
import einops as E


class _ConvBlock(nn.Module):
  dim : int
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(self.dim, (3,3), 1, "SAME", use_bias=False)(x)
    x = nn.GroupNorm(num_groups=None, group_size=1)(x)
    x = nn.relu(x)
    return x


class RecurrentAttention(nn.Module):
  @nn.compact
  def __call__(self, x, num_slot):
    # Init full scope, i.e nothing is explained
    log_scope = jnp.zeros_like(x[...,0:1])
    # Recurrently compute object masks from remaining scope
    log_masks = []
    for _ in range(num_slot-1):
      # Scope is in log scale
      x_scope = jnp.concatenate([x, log_scope], axis=-1) # b h w 4
      b, h, w, c = x_scope.shape
      x1 = _ConvBlock(8)(x_scope)
      x2 = _ConvBlock(16)(jax.image.resize(x1, (b, h//2**1, w//2**1, c), 'nearest'))
      x3 = _ConvBlock(32)(jax.image.resize(x2, (b, h//2**2, w//2**2, c), 'nearest'))
      x4 = _ConvBlock(64)(jax.image.resize(x3, (b, h//2**3, w//2**3, c), 'nearest'))
      x5 = _ConvBlock(64)(jax.image.resize(x4, (b, h//2**4, w//2**4, c), 'nearest'))
      
      # MLP
      z = E.rearrange(x5, 'b h w c -> b (h w c)')
      z = nn.relu(nn.Dense(128)(z))
      z = nn.relu(nn.Dense(128)(z))
      z = nn.relu(nn.Dense(1024)(z))
      z = E.rearrange(z, 'b (h w c) -> b h w c', h=h//2**4, w=w//2**4)

      y5 = _ConvBlock(64)(jnp.concatenate([z, x5], axis=-1))
      y4 = _ConvBlock(64)(jnp.concatenate([jax.image.resize(y5, [b,h//2**3,w//2**3,c], 'nearest'), x4], 3))
      y3 = _ConvBlock(32)(jnp.concatenate([jax.image.resize(y4, [b,h//2**2,w//2**2,c], 'nearest'), x3], 3))
      y2 = _ConvBlock(16)(jnp.concatenate([jax.image.resize(y3, [b,h//2**1,w//2**1,c], 'nearest'), x2], 3))
      y1 = _ConvBlock(8 )(jnp.concatenate([jax.image.resize(y2, [b,h,w,c], 'nearest'), x1], 3))
      alpha = nn.Conv(1, (1,1))(y1)

      log_alpha = jax.nn.log_softmax(jnp.concatenate([alpha, 1-alpha], axis=-1), axis=-1)
      log_masks.append(log_scope + log_alpha[...,0:1])
      log_scope = log_scope + log_alpha[...,1:2]

    # Last mask is the remaining scope
    log_masks.append(log_scope) # Explained the rest
    return log_masks


class SpatialBroadcastDecoder(nn.Module):
  h : int = 64
  w :int = 64

  @nn.compact
  def __call__(self, z):
    # a pair of coordinate channels one for each spatial dimension – ranging from -1 to 1.
    xs = jnp.linspace(-1, 1, self.h+8)
    ys = jnp.linspace(-1, 1, self.w+8)
    coord_map = jnp.meshgrid(xs, ys, indexing='ij')
    coord_map = E.rearrange(coord_map, 'c h w -> h w c')
    coord_map = E.repeat(coord_map, 'h w c -> b h w c', b=z.shape[0])
    z = E.repeat(z, 'b c -> b h w c', h=self.h+8, w=self.w+8)
    z = jnp.concatenate((z, coord_map), axis=-1) # concat in the channel dimension

    z = nn.relu(nn.Conv(32, (3,3), strides=1, padding="VALID")(z))
    z = nn.relu(nn.Conv(32, (3,3), strides=1, padding="VALID")(z))
    z = nn.relu(nn.Conv(32, (3,3), strides=1, padding="VALID")(z))
    z = nn.relu(nn.Conv(32, (3,3), strides=1, padding="VALID")(z))
    z =         nn.Conv(4 , (1,1), strides=1, padding="VALID")(z)
    return z

class cVAE(nn.Module):
  z_dim : int = 16
  beta : float = 0.5

  @nn.compact
  def __call__(self, x):
    # x = image + mask
    x = nn.relu(nn.Conv(32, (3,3), strides=(2,2), padding="SAME")(x))
    x = nn.relu(nn.Conv(32, (3,3), strides=(2,2), padding="SAME")(x))
    x = nn.relu(nn.Conv(64, (3,3), strides=(2,2), padding="SAME")(x))
    x = nn.relu(nn.Conv(64, (3,3), strides=(2,2), padding="SAME")(x))
    x = E.rearrange(x, 'b h w c -> b (h w c)')
    x = nn.relu(nn.Dense(256)(x))
    z = nn.Dense(self.z_dim*2)(x)

    # The MLP output parameterises the μ and log σ of a 16-dim Gaussian latent posterior.
    mean = z[..., :self.z_dim]
    logvar = z[..., self.z_dim:]
    q_zx = dist.MultivariateNormalDiag(mean, jnp.exp(logvar/2)) # q(z|x)
    z = q_zx.sample(seed=self.make_rng("qz"))

    x_rec = SpatialBroadcastDecoder(h=64, w=64)(z)
    return q_zx, x_rec


class MONet(nn.Module):
  num_slot : int
  beta : float = 0.5
  gamma : float = 0.5

  @nn.compact
  def __call__(self, x):
    log_masks = RecurrentAttention()(x, self.num_slot)
    log_masks = E.rearrange(log_masks, 's b h w 1 -> b s h w 1')
    x_slots = E.repeat(x, 'b h w c -> b s h w c', s=self.num_slot)
    x_masks = jnp.concatenate([x_slots, log_masks], axis=-1) # b s h w 4

    x_masks = E.rearrange(x_masks, 'b s h w c -> (b s) h w c')
    q_z, x_masks_rec = cVAE()(x_masks)
    x_masks_rec = E.rearrange(x_masks_rec, '(b s) h w c -> b s h w c', s=self.num_slot)

    # 3 RGB channels for the means of the image components xk and
    x_rec = jax.nn.sigmoid(x_masks_rec[...,0:3])
    # 1 for the logits used for the softmax operation
    # to compute the reconstructed attention masks  mk.
    log_masks_rec = x_masks_rec[...,3:]

    return q_z, (x_slots, x_rec), (log_masks, log_masks_rec)



def _test():
  inputs = jnp.zeros([10,64,64,3])
  model = MONet(num_slot=5)
  rngs = {'params': jax.random.PRNGKey(0), "qz": jax.random.PRNGKey(1)}
  params = model.init(rngs, inputs)
  out = model.apply(params, inputs, rngs={"qz":jax.random.PRNGKey(1)})


if __name__ == "__main__":
  _test()

