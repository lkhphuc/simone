import torch
import torchvision
import pytorch_lightning as pl
import einops as E

class GenerateCallback(pl.Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)[:5]
            with torch.no_grad():
                pl_module.eval()
                rec_x, masks, object_posterior, frame_posterior = pl_module(input_imgs)
                # x_slots, masks_logit, q_z, x_rec, masks_logit_rec = pl_module(input_imgs)
                # q_z, x_rec = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            x_cmb = E.rearrange(torch.sum(rec_x*masks, 1), 'b t c h w -> (b t) c h w')
            input_imgs = E.rearrange(input_imgs, 'b t c h w -> (b t) c h w')
            # x_rec = E.rearrange(x_rec, 'b s c h w -> (s b) c h w', s=6)
            # segmap = masks_to_segmap(masks_logit)
            imgs = torch.cat([input_imgs, x_cmb,], dim=0)
            grid = torchvision.utils.make_grid(imgs, nrow=5, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


COLOR_CODES = [(255,0,0),    # background
               (127,0,0),    # hair
               (255,255,0),  # skin
               (0,0,255),    # eyes
               (0,255,255),  # nose
               (0,255,0)]    # mouth

def masks_to_segmap(masks_logit):
    masks = torch.softmax(masks_logit, dim=1)
    segmaps = []
    for color in range(3):
        segmap = torch.argmax(masks, dim=1)
        for slot in range(masks.shape[1]):
            segmap[segmap==slot] = COLOR_CODES[slot][color]
        segmaps.append(segmap)
    segmaps = torch.cat(segmaps, dim=1)
    return segmaps


def _test():
    log_masks = torch.randn([2, 6, 1, 64, 64])
    segmap = masks_to_segmap(log_masks)
    import streamlit as st
    st.image(segmap.numpy().transpose([0,2,3,1]))

if __name__ == "__main__":
    _test()
