import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as E

class UNet(nn.Module):
    def _make_block(self, dim:int):
        return nn.Sequential(
            nn.LazyConv2d(dim, (3,3), 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(True),
        )
    def __init__(self):
        super().__init__()
        self.encoder1 = self._make_block(8)
        self.encoder2 = self._make_block(16)
        self.encoder3 = self._make_block(32)
        self.encoder4 = self._make_block(64)
        self.encoder5 = self._make_block(64)

        self.mlp = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(inplace=True),
            nn.LazyLinear(128),
            nn.ReLU(inplace=True),
            nn.LazyLinear(1024), # To match x5=64*8*8
            #NOTE how to infer this dim based on input
            nn.ReLU(inplace=True),
        )

        self.decoder5 = self._make_block(64)
        self.decoder4 = self._make_block(64)
        self.decoder3 = self._make_block(32)
        self.decoder2 = self._make_block(16)
        self.decoder1 = self._make_block(8)
        self.decoder0 = nn.LazyConv2d(1, (1,1))

    def forward(self, x, log_scope):
        # Scope is in log scale
        x_scope = torch.cat([x, log_scope], dim=1) # b 4 h w
        x1 = self.encoder1(x_scope)
        x2 = self.encoder2(F.interpolate(x1, scale_factor=1/2))
        x3 = self.encoder3(F.interpolate(x2, scale_factor=1/2))
        x4 = self.encoder4(F.interpolate(x3, scale_factor=1/2))
        x5 = self.encoder5(F.interpolate(x4, scale_factor=1/2))
        
        z = self.mlp(E.rearrange(x5, 'b c h w -> b (c h w)'))
        z = E.rearrange(z, 'b (c h w) -> b c h w', h=x5.shape[2], w=x5.shape[3])

        y5 = F.interpolate(self.decoder5(torch.cat([z, x5], 1)), scale_factor=2)
        y4 = F.interpolate(self.decoder4(torch.cat([y5, x4],1)), scale_factor=2)
        y3 = F.interpolate(self.decoder3(torch.cat([y4, x3],1)), scale_factor=2)
        y2 = F.interpolate(self.decoder2(torch.cat([y3, x2],1)), scale_factor=2)
        y1 = self.decoder1(torch.cat([y2, x1], dim=1))
        alpha = self.decoder0(y1)
        return alpha


class RecurrentAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()

    def forward(self, x, num_slot):
        # Init full scope, i.e nothing is explained
        log_scope = torch.ones_like(x[:, 0:1]).log()
        # Recurrently compute object masks from remaining scope
        log_masks = []
        for _ in range(num_slot-1):
            alpha = self.unet(x, log_scope)
            log_alpha = torch.log_softmax(torch.cat([alpha, 1-alpha], dim=1),dim=1)
            log_masks.append(log_scope + log_alpha[:, 0:1])
            log_scope = log_scope + log_alpha[:, 1:2]
        # Last mask is the remaining scope
        log_masks.append(log_scope) # Explained the rest
        return log_masks


def _test():
    import streamlit as st
    model = RecurrentAttention()
    batch = torch.randn([2,3,64,64])
    masks = model(batch, 5)
    masks = torch.stack(masks).exp()

    masks0 =masks[:,0].detach().numpy().transpose([0,2,3,1])
    st.image(masks0)

if __name__ == "__main__":
    _test()
