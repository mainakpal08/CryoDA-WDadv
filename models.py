import torch.nn as nn
import torch
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Adjust_channel(nn.Module):
    def forward(self, x):
        ch = 1
        return x.reshape(x.shape[0], ch, x.shape[1], x.shape[2], x.shape[3])


class CLFR(nn.Module):
    def __init__(self, opt):
        super(CLFR, self).__init__()
        self.conv1 = self._make_conv_block(1, 8, k=5, pool_k=1, pool_s=1)
        self.conv2 = self._make_conv_block(8, 16, k=5, pool_k=2, pool_s=2)
        self.conv3 = self._make_conv_block(16, 32, k=4, pool=False)
        self.conv4 = self._make_conv_block(32, 64, k=4, pool_k=2, pool_s=2)
        self.conv5 = self._make_conv_block(64, opt.fet_size, k=2, pool_k=2, pool_s=2)
        self.hidden = self._make_lin_block(opt.fet_size, opt.h, r=0.5)


        self.feature_extractor = nn.Sequential(
            Adjust_channel(),
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            Flatten(),
        )
        self.classify = nn.Sequential(
            self.hidden,
            nn.Linear(opt.h, opt.numclass),
        )


    def _make_conv_block(self, inp, oup, k, pool=True, pool_k=None, pool_s=None):
        if pool:
            conv_block = nn.Sequential(
                nn.Conv3d(inp, oup, kernel_size=k),
                nn.ReLU(True),
                nn.MaxPool3d(pool_k, pool_s),
            )

        else:
            conv_block = nn.Sequential(
                nn.Conv3d(inp, oup, kernel_size=k),
                nn.ReLU(True),
            )            

        return conv_block


    def _make_lin_block(self, in_dim, out_dim, r):
        lin_block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(True),
            nn.Dropout(p=r)
        )

        return lin_block


    def forward(self, x):
        feature = self.feature_extractor(x)
        logits = self.classify(feature)

        return logits

    
class CRITIC(nn.Module):
    def __init__(self, opt):
        super(CRITIC, self).__init__()
        self.critic_block = nn.Sequential(
            nn.Linear(opt.fet_size, 50),
            nn.ReLU(True),
            nn.Linear(50, 20),
            nn.ReLU(True),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        critic = self.critic_block(x)

        return critic

