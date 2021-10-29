import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F
import einops as E
from positional_encodings import PositionalEncoding3D


class Encoder(nn.Module):
    def __init__(self, K=16, z_dim=32, conv_dim=128):
        super().__init__()
        # input shape is 64x64
        # I, J : latent's spatial dimension height and width
        # k: slot dimension
        # Input x: b t 64 64 3
        self.K = K
        self.z_dim = z_dim
        self.conv = nn.Sequential(
            nn.LazyConv2d(conv_dim, kernel_size=(3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(conv_dim, kernel_size=(3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(conv_dim, kernel_size=(3,3), stride=2, padding=1),
            nn.ReLU(),
        ) # b t 8 8 128

        tf_layer1 = nn.TransformerEncoderLayer(d_model=conv_dim, nhead=4, dim_feedforward=256, batch_first=True)
        tf_layer2 = nn.TransformerEncoderLayer(d_model=conv_dim, nhead=4, dim_feedforward=256, batch_first=True)
        norm1 = nn.LayerNorm(conv_dim)
        norm2 = nn.LayerNorm(conv_dim)
        self.transformer1 = nn.TransformerEncoder(tf_layer1, 3, norm=norm1)
        self.transformer2 = nn.TransformerEncoder(tf_layer2, 3, norm=norm2) # NOTE: are the weights shared

        self.mlp_frame  = nn.LazyLinear(z_dim*2)
        self.mlp_object = nn.LazyLinear(z_dim*2)
        self.pos = PositionalEncoding3D(conv_dim)

    def forward(self,x):
        t = x.shape[1] # b t h w 3 
        x = E.rearrange(x, 'b t c i j -> (b t) c i j')
        z = self.conv(x) # b t i j c
        del(x)

        # TODO absolute positional embedding
        z = E.rearrange(z, '(b t) c i j -> b t i j c', t=t)
        z = self.pos(z)

        z = E.rearrange(z, 'b t i j c -> b (t i j) c')
        z = self.transformer1(z)

        # Sum Pooling to get K slots
        z = E.rearrange(z, 'b (t i j) c -> b t i j c', t=t, i=8, j=8)
        z = E.reduce(z, 'b t (i i2) (j j2) c -> b t i j c', 'sum', i2=2, j2=2)

        z = E.rearrange(z, 'b t i j c -> b (t i j) c', t=t)
        z = self.transformer2(z)
        z 

        z = E.rearrange(z, 'b (t k) c -> b k t c', t=t, k=self.K)
        object_params = self.mlp_object(E.reduce(z, 'b k t c -> b k c', 'mean'))
        object_posterior = dists.Normal(object_params[:,:,:self.z_dim], object_params[:,:,self.z_dim:].exp())
        frame_params = self.mlp_frame(E.reduce(z, 'b k t c -> b t c', 'mean'))
        frame_posterior = dists.Normal(frame_params[:,:,:self.z_dim], frame_params[:,:,self.z_dim:].exp())
        return object_posterior, frame_posterior


class SIMONe(pl.LightningModule):
    def __init__(self, k=16, h=64, w=64, alpha=1, beta_o=0.5, beta_f=0.5):
        super().__init__()
        self.k = k
        self.h = h
        self.w = w
        self.alpha = alpha
        self.beta_o = beta_o
        self.beta_f = beta_f
        conv_dim=128

        self.encoder = Encoder(z_dim=32) # Inference network
        # Pixel-wise MLP decoder with 1x1 Conv
        self.decoder = nn.Sequential(
            nn.LazyConv2d(conv_dim, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.LazyConv2d(conv_dim, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.LazyConv2d(conv_dim, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.LazyConv2d(conv_dim, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.LazyConv2d(4, kernel_size=(1,1), stride=1),
        )
        # a pair of coordinate channels one for each spatial dimension – ranging from -1 to 1.
        xs = torch.linspace(-1, 1, self.h)
        ys = torch.linspace(-1, 1, self.w)
        xb, yb = torch.meshgrid(xs, ys, indexing='ij')
        self.register_buffer("coord_map", E.rearrange([xb,yb], 'c h w -> c h w'))

    def forward(self, x):
        b, t, c, h, w = x.shape
        coord_map = E.repeat(self.coord_map, 'c h w -> b k t h w c', b=b, k=self.k, t=t)
        timestep = E.repeat(torch.arange(0, t), 't -> b k t h w 1', b=b, k=self.k, h=h, w=w).to(self.device)

        object_posterior, frame_posterior = self.encoder(x)

        object_latents = object_posterior.rsample(torch.Size([t, h, w]))
        object_latents = E.rearrange(object_latents, 't h w b k c -> b k t h w c')
        frame_latents = frame_posterior.rsample(torch.Size([self.k, h, w]))
        frame_latents = E.rearrange(frame_latents, 'k h w b t c -> b k t h w c')
        latents = torch.cat([object_latents, frame_latents, coord_map, timestep], dim=-1)

        # TODO strided slice of inputs
        rec_x_masks = self.decoder(E.rearrange(latents, 'b k t h w c -> (b k t) c h w'))

        rec_x_masks = E.rearrange(rec_x_masks, '(b k t) c h w -> b k t c h w', b=b, k=self.k, t=t)
        rec_x, masks = rec_x_masks[:,:,:,:3,:,:], rec_x_masks[:,:,:,3:,:,:]
        masks = F.softmax(masks, dim=1) # softmax along the K object dims

        return rec_x, masks, object_posterior, frame_posterior


    def training_step(self, x, batch_idx):
        rec_x, masks, object_posterior, frame_posterior = self.forward(x)
        rec_x_full = torch.sum(rec_x*masks, dim=1) # Full frame equals mixture over slots
        del(rec_x, masks)

        x_prior = dists.Normal(torch.sigmoid(rec_x_full), torch.ones_like(rec_x_full)*0.08)
        nll = -x_prior.log_prob(x)
        nll = E.reduce(nll, 'b t c h w -> b t h w', 'sum').mean() # Mean over b+t+h+w

        object_prior = dists.Normal(0., 1.)
        object_kl = dists.kl_divergence(object_posterior, object_prior)
        object_kl = E.reduce(object_kl, 'b k c -> b k', 'sum').mean() # Mean over b+k

        frame_prior = dists.Normal(0.0, 1.0)
        frame_kl = dists.kl_divergence(frame_posterior, frame_prior)
        frame_kl = E.reduce(frame_kl, 'b t c -> b t', 'sum').mean() # Mean over b+t

        loss = self.alpha*nll + self.beta_o*object_kl + self.beta_f*frame_kl

        self.log("loss/train", loss)
        self.log("loss/nll", self.alpha*nll)
        self.log("loss/latent_o", self.beta_o*object_kl)
        self.log("loss/latent_f", self.beta_f*frame_kl)
        return loss

    def validation_step(self, x, batch_idx):
        rec_x, masks, object_posterior, frame_posterior = self.forward(x)
        rec_x_full = torch.sum(rec_x*masks, dim=1) # Full frame equals mixture over slots
        del(rec_x, masks)

        x_prior = dists.Normal(torch.sigmoid(rec_x_full), torch.ones_like(rec_x_full)*0.08)
        nll = -x_prior.log_prob(rec_x_full)
        nll = E.reduce(nll, 'b t c h w -> b t h w', 'sum').mean() # Mean over b+t+h+w

        object_prior = dists.Normal(
            torch.zeros_like(object_posterior.loc), torch.ones_like(object_posterior.scale))
        object_kl = dists.kl_divergence(object_prior, object_posterior)
        object_kl = E.reduce(object_kl, 'b k c -> b k', 'sum').mean() # Mean over b+k

        frame_prior = dists.Normal(
            torch.zeros_like(frame_posterior.loc), torch.ones_like(frame_posterior.scale))
        frame_kl = dists.kl_divergence(frame_prior, frame_posterior)
        frame_kl = E.reduce(frame_kl, 'b t c -> b t', 'sum').mean() # Mean over b+t

        loss = self.alpha*nll + self.beta_o*object_kl + self.beta_f*frame_kl

        return loss


    def validation_epoch_end(self, outputs) -> None:
        loss = torch.tensor(outputs).mean()
        self.log("loss/val", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=20e-5)
        return optimizer


def _test():
    model = SIMONe().cuda()
    inputs = torch.ones([1,10,3,64,64]).cuda()
    out = model(inputs)
    print(out.shape)


if __name__ == "__main__":
    _test()
