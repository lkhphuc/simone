import pytorch_lightning as pl
import torch
import torch.distributions as dists
import torch.nn.functional as F
import einops as E

from .cvae import cVAE
from .attn import RecurrentAttention

class MONet(pl.LightningModule):

    def __init__(self, num_slot, beta=0.5, gamma=0.5):
        super().__init__()
        self.cvae = cVAE()
        self.attn = RecurrentAttention()
        self.num_slot = num_slot
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        log_masks = self.attn(x, self.num_slot)
        log_masks = E.rearrange(log_masks, 's b 1 h w -> b s 1 h w')
        x_slots = E.repeat(x, 'b c h w -> b s c h w', s=self.num_slot)
        x_masks = torch.cat([x_slots, log_masks], dim=2) # b s 4 h w

        x_masks = E.rearrange(x_masks, 'b s c h w -> (b s) c h w')
        q_z, x_masks_rec = self.cvae(x_masks)
        x_masks_rec = E.rearrange(x_masks_rec, '(b s) c h w -> b s c h w', s=self.num_slot)

        # 3 RGB channels for the means of the image components xk and
        x_rec = torch.sigmoid(x_masks_rec[:, :, 0:3])
        # 1 for the logits used for the softmax operation
        # to compute the reconstructed attention masks  mk.
        log_masks_rec = x_masks_rec[:, :, 3:]

        return x_slots, log_masks, q_z, x_rec, log_masks_rec

    def training_step(self, batch, batch_idx):
        x_slots, log_masks, q_z, x_rec, log_masks_rec = self.forward(batch)
        del(batch)

        scale = torch.ones_like(x_rec, device=self.device) * 0.11
        scale[:,0,:,:,:] = 0.09  # different scale for 1st slot
        p_x = dists.Normal(x_rec, scale)
        nll = -p_x.log_prob(x_slots) * log_masks.exp() # Weighted by the masks
        nll = E.reduce(nll, 'b s c h w -> b', 'sum').mean()

        p_z = dists.Normal(0., 1.)
        latent_loss = dists.kl_divergence(q_z, p_z)
        latent_loss = E.reduce(latent_loss, '(b s) c -> b', 'sum', s=self.num_slot).mean()

        q_c = dists.Categorical(logits=E.rearrange(log_masks,     'b s c h w -> b c h w s'))
        p_c = dists.Categorical(logits=E.rearrange(log_masks_rec, 'b s c h w -> b c h w s'))
        mask_loss = E.reduce(dists.kl_divergence(q_c, p_c), 'b c h w -> b', 'sum').mean()

        loss = nll + self.beta*latent_loss + self.gamma*mask_loss

        self.log("loss/train", loss)
        self.log("loss/nll", nll)
        self.log("loss/latent", latent_loss)
        self.log("loss/mask", mask_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_slots, log_masks, q_z, x_rec, log_masks_rec = self.forward(batch)
        del(batch)

        scale = torch.ones_like(x_rec, device=self.device) * 0.11
        scale[:,0,:,:,:] = 0.09  # different scale for 1st slot
        p_x = dists.Normal(x_rec, scale)
        nll = -p_x.log_prob(x_slots) + log_masks # Weighted by the masks
        nll = E.reduce(nll, 'b s c h w -> b', 'sum').mean()

        p_z = dists.Normal(0., 1.)
        latent_loss = dists.kl_divergence(q_z, p_z)
        latent_loss = E.reduce(latent_loss, '(b s) c -> b', 'sum', s=self.num_slot).mean()

        q_c = dists.Categorical(logits=E.rearrange(log_masks,     'b s c h w -> b c h w s'))
        p_c = dists.Categorical(logits=E.rearrange(log_masks_rec, 'b s c h w -> b c h w s'))
        mask_loss = E.reduce(dists.kl_divergence(q_c, p_c), 'b c h w -> b', 'sum').mean()

        loss = nll + self.beta*latent_loss + self.gamma*mask_loss
        return loss

    def validation_epoch_end(self, outputs) -> None:
        loss = torch.tensor(outputs).mean()
        self.log("loss/val", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-4)
        return optimizer

def _test():
    import streamlit as st

    model = MONet(num_slot=6)
    input = torch.randn((2, 3, 64, 64))
    # scope = torch.rand((1, 1, 64, 64))
    out = model(input)
    print(out.shape)
    out = out.detach().cpu().numpy().transpose((0,2,3,1))
    st.image(E.rearrange(out, 'b h w (c c1) -> (b c) h w c1', c1=1))

    out = out.sum(-1, keepdims=True)
    st.image(1-(out.sum(-1,keepdims=True)/out.max()))

if __name__ == "__main__":
    _test()
