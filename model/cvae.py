import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as E
import torch.distributions as dists
import pytorch_lightning as pl

class SpatialBroadcastDecoder(nn.Module):
    def __init__(self,spatial_size=64, out_dim=4):
        super().__init__()
        self.h = spatial_size + 4*2 # each conv no padding -2
        self.w = spatial_size + 4*2
        self.layers = nn.Sequential(
            nn.LazyConv2d(32, (3,3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(32, (3,3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(32, (3,3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(32, (3,3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(out_dim, (1,1), stride=1, padding=0)
        )

        # a pair of coordinate channels one for each spatial dimension – ranging from -1 to 1.
        ys = torch.linspace(-1, 1, self.h)
        xs = torch.linspace(-1, 1, self.w)
        xb, yb = torch.meshgrid(ys, xs) 
        self.register_buffer("coord_map", E.rearrange([xb,yb], 'c h w -> c h w'))

    def forward(self, z):
        # The input to the broadcast decoder is a spatial tiling of zk
        z = E.repeat(z, 'b c -> b c h w', h=self.h, w=self.w)
        coord_map = E.repeat(self.coord_map, 'c h w -> b c h w', b=z.shape[0])
        z_sb = torch.cat((z, coord_map), dim=1) # concat in the channel dimension
        return self.layers(z_sb)


class cVAE(pl.LightningModule):
    def __init__(self, z_dim:int=16, out_dim=4, beta=0.5):
        super().__init__()
        self.z_dim = z_dim
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.LazyConv2d(32, (3,3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(32, (3,3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(64, (3,3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(64, (3,3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            )
        self.mlp = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.LazyLinear(z_dim*2),
        )
        self.decoder = SpatialBroadcastDecoder(spatial_size=64, out_dim=out_dim)

    def forward(self, x_mask):
        z = self.encoder(x_mask)
        z = E.rearrange(z, 'b c h w -> b (c h w)')
        z = self.mlp(z)

        # The MLP output parameterises the μ and log σ of a 16-dim Gaussian latent posterior.
        mean = z[:, :self.z_dim]
        logvar = z[:, self.z_dim:]

        # Reparameterize latent
        q_z = dists.Normal(mean, torch.exp(logvar*0.5))
        z = q_z.rsample()

        x_mask_rec = self.decoder(z) # output into 4 channels:

        return q_z, x_mask_rec

    def training_step(self, batch, batch_idx):
        q_z, x_mask_rec = self.forward(batch)
        
        p_x = dists.Normal(torch.sigmoid(x_mask_rec), torch.ones_like(x_mask_rec)*0.1)
        nll = -p_x.log_prob(batch).sum([1,2,3]).mean()

        p_z = dists.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
        kl = dists.kl_divergence(q_z, p_z).sum(1).mean(0)

        loss = nll + self.beta*kl

        # print(f"{nll.item()=}, {kl.item()=}, {loss.item()=}")
        self.log("loss/latent", kl)
        self.log("loss/nll", nll)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        q_z, x_mask_rec = self.forward(batch)
        
        p_z = dists.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
        kl = dists.kl_divergence(q_z, p_z).sum(1).mean(0)

        p_x = dists.Normal(torch.sigmoid(x_mask_rec), torch.ones_like(x_mask_rec)*0.1)
        nll = -p_x.log_prob(batch).sum([1,2,3]).mean()
        loss = nll + self.beta*kl
        return loss

    def validation_epoch_end(self, outputs):
        loss = torch.Tensor(outputs).mean()
        self.log("loss/val", loss)

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=1e-4)





def test():
    model = cVAE(out_dim=3)
    x_masks = torch.randn([2,3,64,64])
    z, mean, logvar, x_mask_rec = model(x_masks)

if __name__ == "__main__":
    test()
