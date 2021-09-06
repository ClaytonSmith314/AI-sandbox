
import torch
import torch.nn as nn
import numpy as np



class ResConvBlock(nn.Module):
    """
    nn block based on paper-
        Deep Residual Learning for Image Recognition
    arXiv:1512.03385, https://arxiv.org/abs/1512.03385
    """
    def __init__(self, input_channels, hidden_channels, output_channels,
                 kernal_size=(3,3), layer1_padding=(1,1), layer2_padding=(1,1)):
        super(ResConvBlock, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernal_size, padding=layer1_padding),
            nn.BatchNorm2d(num_features=hidden_channels), #TODO: figure out if 1d or 2d should be used
            nn.ReLU(),

            nn.Conv2d(hidden_channels, output_channels, kernal_size, padding=layer2_padding),
            nn.BatchNorm2d(num_features=output_channels),
        )

        self.relu = nn.ReLU()
        self.has_channel_diff = (input_channels != output_channels)
        if self.has_channel_diff:
            self.conv1x1 = nn.Conv2d(input_channels, output_channels, (1,1))

    def forward(self, x):
        if self.has_channel_diff:
            return self.relu(self.conv1x1(x) + self.conv_stack(x))
        else:
            return self.relu(x + self.conv_stack(x))



class MixerBlock(nn.Module):
    """
    nn block based on paper-
        MLP-Mixer: An all-MLP Architecture for Vision
    arXiv:2105.01601, https://arxiv.org/abs/2105.01601
    """
    def __init__(self, channels, patches,
                 hidden_dim_1=None, hidden_dim_2=None):
        super(MixerBlock, self).__init__()

        # Note: this is not from the paper. hidden dimensions can be chosen.
        if hidden_dim_1 is None:
            hidden_dim_1 = patches
        if hidden_dim_2 is None:
            hidden_dim_2 = channels


        self.mlp1 = nn.Sequential(
            nn.Linear(in_features=patches, out_features=hidden_dim_1),
            nn.ReLU(),  #note: in paper, gelu is used instead of relu
            nn.Linear(in_features=hidden_dim_1, out_features=patches)
        )

        self.lnorm_mlp2 = nn.Sequential(
            nn.LayerNorm(channels),

            nn.Linear(in_features=channels, out_features=hidden_dim_2),
            nn.ReLU(),  # note: in paper, gelu is used instead of relu
            nn.Linear(in_features=hidden_dim_2, out_features=channels)
        )

        self.channels = channels
        self.patches = patches

    def forward(self, x: torch.Tensor):
        assert x.size(0) == self.patches
        x = x.view(self.patches, self.channels)
        x_t = torch.transpose(x, -1, -2)
        x_t = self.mlp1(x_t)
        x = x + torch.transpose(x_t, -1, -2)
        x = x + self.lnorm_mlp2(x)
        return x.view(self.patches, 1, self.channels)




class LinearReduction(nn.Module):
    """
    block based loosely on Mixer but with an aim to quickly reduce dimensionality of 2d data
    """
    def __init__(self, channels, patches, new_channels, new_patches,
                 hidden_dim_1=None, hidden_dim_2=None):
        super(LinearReduction, self).__init__()

        # Note: this is not from the paper. hidden dimensions can be chosen.
        if hidden_dim_1 is None:
            hidden_dim_1 = patches
        if hidden_dim_2 is None:
            hidden_dim_2 = channels

        self.lin1 = nn.Linear(in_features=patches, out_features=new_patches)

        self.lin2 = nn.Linear(in_features=channels, out_features=new_channels)

        self.channels = channels
        self.new_channels = new_channels
        self.patches = patches
        self.new_patches = new_patches

    def forward(self, x: torch.Tensor):
        assert x.size(0) == self.patches
        x = x.view(self.patches, self.channels)
        x_t = torch.transpose(x, -1, -2)
        x_t = self.lin1(x_t)
        x = torch.transpose(x_t, -1, -2)
        x = self.lin2(x)
        return x.view(self.new_patches, 1, self.new_channels)


class TransformerBlock(nn.Module):
    """
    nn block based on paper
        Attention Is All You Need
    arXiv:1706.03762, https://arxiv.org/abs/1706.03762

    note: foward has three inputs, however only the queries have residual connections
    """
    #TODO: figure out how to turn on batch first
    def __init__(self, embed_size, num_heads, dropout=0, feed_forward_hidden_dim=2048):
        super(TransformerBlock, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads,
                                         dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, feed_forward_hidden_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden_dim, embed_size)
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, queries, keys=None, values=None):
        #input shape is (L, N, E). For our use, N = 1
        if keys is None:
            keys = queries
            values = queries
        mha_out, _ = self.mha(queries, keys, values)
        mha_out = self.dropout(self.norm1(queries + mha_out))
        forward_out = self.feedforward(mha_out)
        output = self.dropout(self.norm2(mha_out + forward_out))
        return output




class SpatialEmbed(nn.Module):
    """
    adds cosine embeding to image, which can then be flattened across width*height
    to make an attention tensor
    """
    def __init__(self, channels, height, width, xy_embed_len):
        # size should be [channels, width*height]
        super(SpatialEmbed, self).__init__()

        assert xy_embed_len*2 <= channels

        # assert xy_embed_len >= width and xy_embed_len >= height
        embed = np.zeros(shape=(channels, height, width))
        for x in range(0, width):
            for y in range(0, height):
                for c in range(0, xy_embed_len):
                    embed[c, y, x] = np.cos(x/width*np.pi*(c+1)) #TODO: make sure this is good enough embeding
                for c in range(xy_embed_len, 2*xy_embed_len):
                    embed[c, y, x] = np.cos(y/height*np.pi*(c+1))

        self.embed = torch.from_numpy(embed)
        # note: embed is not registered as a parameter

    def forward(self, image):
        return image + self.embed




