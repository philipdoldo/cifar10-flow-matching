import torch
import torch.nn as nn
import torch.functional as F
from dataclasses import dataclass
import yaml
import math


class SinusoidalEmbedding(nn.Module):

    def __init__(self, d=256, base=10000):
        super().__init__()
        if d % 2 != 0:
            raise ValueError(f"Embedding dimension {d=} must be a multiple of 2")
        self.d = d # embedding dimension
        self.base = base # e.g. 10000
    
    def forward(self, t):
        """
        `t` has shape (B, 1) --- batch of times in [0, 1]
        `t` gets mapped to a batch of embeddings of dimension d --- output has shape (B, d)
        """
        i = torch.arange(self.d // 2, device=t.device()) # shape (d//2)
        freqs = 1/self.base ** ((2 * i) / self.d) # shape (d//2)
        angles = freqs * t # (d//2) * (B, 1) --> broadcasts to (B, d//2)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)
    

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim=None, bias=False):
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * input_dim

        W1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        W2 = nn.Linear(hidden_dim, input_dim, bias=bias)

    def forward(self, x):
        """
        `x` has shape (B, d) where B is batch size and d is embedding dimension
        """
        x = self.W1(x)
        x = F.relu(x).square()
        x = self.W2(x)
        return x


class TimeAndClassEmbedding(nn.Module):

    def __init__(self, d, hidden_dim=None, num_classes=10, base=10000):
        """
        `d` is the embedding dimension for time/class
        `hidden_dim` is the optional hidden dimension in the MLP, defaults to 4*d
        """
        super().__init__()
        self.time_embedding = SinusoidalEmbedding(d=d, base=base)
        self.class_embedding = nn.Embedding(num_classes+1, d) # +1 for null class for classifier-free guidance (CFG)
        self.mlp = MLP(input_dim=d, hidden_dim=hidden_dim)
    
    def forward(self, t, y):
        """
        `t` has shape (B, 1) --- batch of times which are scalars in [0, 1]
        `y` has shape (B, 1) --- batch of class labels which are in {0, ..., 10}. Class labels in {0, ..., 9} correspond to MNIST 
        digits and a class label of 10 corresponds to the empty class label, which is used for classifier-free guidance (CFG)

        output a batch of embedding vectors: shape (B, d)
        """
        class_emb = self.class_embedding(y) # shape (B, d)
        time_emb = self.time_embedding(t) # shape (B, d)
        embeddings = self.mlp(class_emb + time_emb) # shape (B, d)
        return embeddings # shape (B, d)


class GroupNorm32(nn.GroupNorm):
    """
    Taken from https://github.com/KellyYutongHe/cmu-10799-diffusion/blob/main/src/models/blocks.py#L69

    GroupNorm is initialized with `num_groups`, `num_channels`, `eps` (default 1e-5), and `affine` (default True)

    GroupNorm takes a tensor with shape e.g. (B, C, H, W) as input (only the first two dimensions matter) where C must
    be equal to `num_channels` and C/num_groups (the group size) must be an integer. For a fixed batch and a group of channels,
    there are (C/num_group)*H*W values and the sample mean mu(b, g) and variance sigma^2(b, g) are computed where b is the batch
    and g is the group of channels. Every entry in the group is standardized according to these values (i.e. by doing
    (x-mean)/(sqrt(variance + eps), roughly speaking). This is performed across all batches and groups. When `affine` is true,
    there are two learnable vectors of shape (C,), one which scales the result and one which shifts it, where the scale and shift
    are potentially different for each channel. The scale is initialized to all 1's and the shift is initialized to all 0's. 

    GroupNorm32 does GroupNorm but with float32 precision since it is claimed that diffusion models can be sensitive to numerical
    precision during normalization. It casts to float32 and then back to the original precision. It would be interesting to test
    how much this actually matters.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def rmsnorm(x):
    """
    `x` has shape (B, C, H, W) and the RMS (which is just 2-norm scaled by 1/sqrt(d) in R^d) is computed for every vector of channels
    No learnable parameters. I want to try rmsnorm since I used it in language models. 
    """
    rms = x.pow(2).mean(dim=1, keepdim=True).sqrt() # shape (B, 1, H, W)
    return (x / (rms + 1e-8))


def rmsnorm32(x):
    """
    rmsnorm but now with float32 precision just in case that ends up being important for stability
    """
    rms = x.float().pow(2).mean(dim=1, keepdim=True).sqrt()
    return (x / (rms + 1e-8)).type(x.dtype)



class Downsample(nn.Module):
    """
    Downsampling layer that halves the spatial dimensions (height and width)

    Input has shape (B, C, H, W) where C is the number of input channels and the output should have shape (B, C, ceil(H/2), ceil(W/2))
    Naively, we don't assume that H and W are even because many images have spatial dimensions that are not even divisible by 4 or 8 and thus
    we need to accept that downsampling followed by upsampling could actually strictly increase the spatial dimensions and thus we need
    to be careful with our residual connections as they might not have the same spatial dimensions --- however, it seems like in practice
    people actually just resize their image spatial dimensions in a preprocessing step to the nearest power of 2 to avoid this issue. Because
    of this, I will assert that the input spatial dimensions must be even.

    Originally, I was going to use nn.MaxPool2d(kernel_size=2, stride=2) which considers a 2x2 grid (starting in the top-left corner
    with no padding, so that the entire 2x2 grid is fully contained within the image) and consolidates ("pools") the 4 values in the
    2x2 grid into a single value by taking the maximum of those 4 values. Note that the stride of 2 applies both horizontally and
    vertically so no two 2x2 grids will overlap and the width and height will be reduced by a factor of 2 (this assumes width and height
    were both even, of course). This pooling process is performed independently for each channel.

    However, instead of doing 2x2 max pooling with a stride of 2, you can do a 3x3 convolution with a stride of 2, a padding of 1, and
    using the same number of filters as there are input channels (so the output has the same number of channels that the input has) and
    this will also reduce the spatial dimensions by a factor of 2 but now instead of arbitrarily restricting the model to do max pooling,
    we are effectively allowing the model to learn what kind of (potentially complicated) pooling to do. Each filter has shape (C, 3, 3) and
    we are using C filters, so this comes at the cost of adding 9*C^2 parameters, but it might learn a smarter pooling approach. I got this
    idea from https://github.com/KellyYutongHe/cmu-10799-diffusion/blob/main/src/models/blocks.py#L254 which I also took some other minor
    inspirations from here and there.
    """

    def __init__(self, channels):
        """
        `channels` is an integer which is the number of input channels (and output channels)
        """
        super().__init__()
        self.channels = channels
        
        # out_channels is the number of filters, an integer for kernel_size gives a square grid, so kernel_size=3 gives a 3x3 grid
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        `x` has shape (B, C, H, W) where B is batch size, H is height, W is width, and C is channels
        output will have shape
        """
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"{x.shape=}, {H=}, {W=}, but both must be even to avoid spatial dimension mismatches when using the residual connections in the U-Net")
        if C != self.channels:
            raise ValueError(f"{x.shape=}, {C=}, {self.channels=}")
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling layer that doubles the spatial dimensions (height and width)

    Input has shape (B, C, H, W) where C is the number of input channels and the output should have shape (B, C//2, 2*H, 2*W)

    Something I almost forgot was to be sure our convolution halves the number of channels because we'll be concatenating along
    the channel dimension with the output from the encoder via the skip connection, so really this class upsamples AND 


    Originally I was going to do a "transposed convolution" which uses C filters of of shape (C, 2, 2) and for each pixel value
    in a given channel, the pixel value p scales all 4 values in the 2x2 grid for the current value which results in a 2x2 for
    each channel, all C of these 2x2 matrices are summed to obtain a single 2x2 matrix corresponding to this filter. This is then
    performed for all C filters (since we chose to use C filters, the number of output channels matches the number of input channels).
    I was going to use nn.ConvTranspose2d(in_channels=F, out_channels=F//2, kernel_size=2, stride=2) which has a stride of 2 which 
    means that each 2x2 output matrix (which corresponds to each input pixel) would not overlap with any of the other 2x2 output
    matrices. People claim that transposed convolution (a.k.a. "deconvolution") results in checkboard artifacts, but I feel like
    this should only happen when the stride and kernel size are chosen such that the output matrices overlap. In my case, there
    would be no overlap so I wouldn't expect checkboard artifacts. Regardless, I'm going to instead try the nearest-neighbor upsampling
    because I should learn what it is:

    Nearest-neighbor upsampling is actually pretty simple: you call `F.interpolate(x, scale_factor=2, mode='nearest')` and then apply
    a convolution to its output, e.g. `nn.Conv2d(channels, channels, kernel_size=3, padding=1)`. The choice to use a 3x3 convolution 
    seems like it is kind of arbitrary, I am just copying this from https://github.com/KellyYutongHe/cmu-10799-diffusion/blob/main/src/models/blocks.py#L286
    EDIT: I think there was a typo and they meant to do `nn.Conv2d(channels, channels, kernel_size=3, padding=1)` which explains why I
    thought it felt arbitrary at first, turns out it is necessary for halving the channels to make the channel size work with the concatenation.
    
    So what does `F.interpolate(x, scale_factor=2, mode='nearest')` actually do? Suppose x has shape (B, C, H, W), consider a single
    H-by-W grid for a fixed batch and channel. We define H' = round(scale_factor * H) and W' = round(scale_factor * W) (I don't actually
    know how rounding is done, but it isn't important), so the height of our grid went from having indices [0, 1, 2, ..., H-1] to having
    indices [0, 1, 2, ..., H'-1] and in our case H' = 2*H. If we have some height index h' in our new grid, we determine its value by 
    computing `index = round(h' * H/H')` which effectively divides by the scale factor which initially gives a value that might not be
    an integer (and thus not a valid index) and so the `mode='nearest'` argument makes the decision to round it to the nearest (presumably
    valid) integer to determine the index `h` to use in the original grid and then uses the pixel value at (h, w) (assuming w was determined
    in an analogous way for the width) as the value at (h', w') in the new grid. I don't know the precise rounding details and said "presumably
    valid" earlier because if H=10 and H'=20, we'd have height indices [0, 1, ..., 9] and [0, 1, ..., 19] and 19/2 = 9.5 and round(9.5) might
    give 10 which would be an invalid index, so 9 would be the only valid choice --- I'm not worrying about minor details like this. I'm
    guessing that other options for `mode` like 'linear' just linearly interpolate between the two pixel values in the original grid. My
    understanding is that this interpolate operation is performed independently for every channel. 
    """

    def __init__(self, channels):
        """
        `channels` is an integer which is the number of input channels (and output channels)
        """
        super().__init__()
        self.channels = channels
        # The convolution halves the number of channels to account for the concatenation from the skip connection, see Fig. 1 in  https://arxiv.org/pdf/1505.04597
        self.conv = nn.Conv2d(channels, channels//2, kernel_size=3, padding=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"{x.shape=}, {C=}, {self.channels=}")
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class ResNetBlock(nn.Module):
    """
    Partially inspired by https://github.com/KellyYutongHe/cmu-10799-diffusion/blob/main/src/models/blocks.py#L85
    I'm choosing not to use FiLM for now to keep things simple, can always ablate later

    We use 3x3 Conv w/ padding=1 to preserve spatial dimensions (this avoids the cropping that was done in the original U-Net paper)

    z -> Nonlinearity -> Linear -> z'
    x -> Norm -> Nonlinearity -> Conv -> add z' -> Norm -> Nonlinearity -> Dropout -> Conv -> Residual Connection (add x or Conv(x), use the latter if in_channels != out_channels)  
    
    input of shape (B, in_channels, H, W) gets mapped to output of shape (B, out_channels, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int, d: int, dropout: float = 0.0):
        """
        `in_channels` is the number of channels in the input
        `out_channels` is the number of channels in the output
        `d` is the time/class embedding dimension
        `dropout` is the dropout probability
        """
        super().__init__()
        self.in_channels = in_channels

        self.W = nn.Linear(d, out_channels) # projection for time/class embeddings for this specific layer

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual Connection
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # in this case, the point of the convolution is just to adjust the number of channels so we can add it to the output at the end

    def forward(self, x, z):
        """
        `x` has shape (B, C, H, W) is the primary input tensor
        `z` has shape (B, d) and represents the time/class embedding vectors, d is the dimension of these embeddings
        """
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"{x.shape=}, {C=}, {self.in_channels=}")

        # First Conv Block: Norm -> Nonlinearity -> Conv
        h = rmsnorm(x)          # (B, C, H, W)      intentionally keep `x` unmodified for residual connection at the end
        h = F.relu(h).square()  # (B, C, H, W)
        h = self.conv1(h)       # (B, out_channels, H, W)

        # apply nonlinearity and then project time/class embeddings 
        z = F.relu(z).square()  # (B, d)
        z = self.W(z)           # (B, out_channels)
        z = z[:, :, None, None] # (B, out_channels, 1, 1)

        # Add context from time/class embeddings 
        h = h + z               # (B, out_channels, H, W)

        # Second Conv Block: 
        h = rmsnorm(h)          # (B, out_channels, H, W)
        h = F.relu(h).square()  # (B, out_channels, H, W)
        h = self.dropout(h)     # (B, out_channels, H, W)
        h = self.conv2(h)       # (B, out_channels, H, W)

        # Residual Connection
        return h + self.skip_connection(x) # (B, out_channels, H, W)


class SelfAttention(nn.Module):
    """
    - No causal mask
    - Takes (B, C, H, W) as input but will *effectively* treat it like (B, H*W, C) where H*W is the sequence length and C is 
    the "token" embedding dimension, though we'll use a 1x1 convolution on this input to go from (B, C, H, W) to (B, 3*C, H, W)
    and then reshape to (B, H*W, 3*C) to effectively get our q, k, v tensors all in one rather than reshaping our input from
    (B, C, H, W) to (B, H*W, C) and then using three separate linear layers to compute `q, k, v = W_q(x), W_k(x), W_v(x)` 
    because the convolution approach should only have to launch a single kernel rather than three separate kernels for each of
    the linear layers. I don't actually know if this makes a difference in practice, maybe torch.compile() would fuse the three
    linear layers anyway, I have no idea, but the point is that the convolutional approach is also valid and effectively equivalent
    to the three linear layers, so we might as well use it since it seems like it'll probably be faster than the three linear layers. 
    - We also use a 1x1 convolution for the output project because it is equivalent and it skips an extra transpose that would be
    required to do a regular linear layer while still returning the correct output shape
    - I won't use RoPE for now. Seems like ViTs have used 2D RoPE before? maybe test that later
    """
    def __init__(self, channels:int, num_heads: int):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads

        self.qkv_conv = nn.Conv2d(channels, 3*channels, kernel_size=1) # effectively acts like the 3 projection matrices W_q, W_k, and W_v
        self.output_conv = nn.Conv2d(channels, channels, kernel_size=1) # effectively acts like the linear output projection W_o

    
    def forward(self, x):
        """
        `x` has shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"{x.shape=}, {C=}, {self.channels=}")
        if C % self.num_heads != 0:
            raise ValueError(f"{C=}, {self.num_heads=}, {C/self.num_heads=} we need our head_size to be an integer")
        head_size = C // self.num_heads


        qkv = self.qkv_conv(x) # (B, 3*C, H, W)

        qkv = qkv.view(B, 3, self.num_heads, head_size, H * W)  # (B, 3, num_heads, head_size, H*W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, H*W, head_size)

        assert q.shape[-1] == head_size, f"{q.shape=}, {head_size=}"

        # TODO: would apply RoPE to q and k here...

        q, k = rmsnorm(q), rmsnorm(k) # QK norm

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False) # (B, num_heads, H*W, head_dim)
        
        ##### Manual version of scaled_dot_product_attention (no causal mask) would be:
        # att = (q @ k.transpose(3, 2)) * (1.0 / math.sqrt(head_size)) # (B, num_heads, H*W, H*W) <-- (B, num_heads, H*W, head_size) x (B, num_heads, head_size, H*W)
        # att = F.softmax(att, dim=-1) # (B, num_heads, H*W, H*W)
        # y = att @ v # (B, num_heads, H*W, head_dim) <-- (B, num_heads, H*W, H*W) x (B, num_heads, H*W, head_dim)

        y = y.transpose(2, 3) # (B, num_heads, head_dim, H*W)
        y = y.contiguous().view(B, C, H, W)
        return 


@dataclass
class UNetConfig:
    in_channels: int = 3
    model_channels: int = 128
    channel_multipliers: tuple = (1, 2, 4, 8)
    num_blocks: int = 1
    time_embed_dim: int = 128
    dropout: float = 0.0
    num_heads: int = 4

    @classmethod
    def from_yaml(cls, path: str) -> "UNetConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))

def is_power_of_2(n: int):
    """
    returns True if `n` is an integer power of 2 and 0 otherwise
    `&` computes bitwise AND between two integers, e.g. 8 & 7 = 0b1000 & 0b0111 = 0b0000 = 0
    """
    return n > 0 and (n & (n-1) == 0)

class UNet(nn.Module):
    """
    Inspired by https://arxiv.org/abs/1505.04597 but not intended to be a faithful replication of their implementation.

    Based on https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/unet.py#L19 people downsample 3 times with 32x32 images (down to 4x4)

    It seems like people typically rescale images to be square so that H = W and H and W are often chosen to be powers of 2. In this code, I am
    simply going to assume that H = W = 2^n for some positive integer n. 
    """

    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config

        self.channels = config.channels # initial channels in the image
        self.H_init = config.initial_image_height # note: arbitrarily naming after height, could've chosen width since assuming square images
        self.H_min = config.min_image_height # the smallest we allow the image spatial dimensions to get after downsampling
        if not is_power_of_2(self.H_init) or not is_power_of_2(self.H_min) or self.H_min >= self.H_init:
            raise ValueError(f"{self.H_init=}, {self.H_min=}, but need both to be powers of 2 and H_init must be greater than H_min")
        self.num_downsamples = int(math.log2(self.H_init // self.H_min))
        if self.num_downsamples < 1:
            raise ValueError(f"{self.num_downsamples=}, {self.H_init=}, {self.H_min=}")

        self.max_attention_height = config.max_attention_height # the maximum image height H that we will use self-attention layers for (if height is too big, the sequence length of H*W = H^2 is too expensive)
        self.base_channels = config.base_channels # the number of out_channels from the first ResNetBlock
        self.d = config.d # time/class embedding dimension
        self.dropout = config.dropout # dropout probability for resnetblocks

        # Encoder/Down portion of the U-Net
        self.down_blocks = []
        for i in range(self.num_downsamples):
            if i == 0:
                in_channels = self.channels
                out_channels = self.base_channels
            else:
                in_channels = out_channels
                out_channels = 2*in_channels # hardcoding 2 here since people seem to double channels to compensate for downsampling by factor of 2 and we indeed assumed a factor of 2 when downsampling
            block = ResNetBlock(in_channels=in_channels, out_channels=out_channels, d=self.d, dropout=self.dropout)
            self.down_blocks.append(block)
        
        # The Bottleneck/Bottom portion of the U-Net
        in_channels = out_channels
        out_channels = 2*in_channels
        self.bottleneck_conv = ResNetBlock(in_channels=in_channels, out_channels=out_channels, d=self.d, dropout=self.dropout)

        # The Decoder/Up portion of the U-Net
        self.up_blocks = []
        for i in range(self.num_downsamples): # we have the same number of upsamples as we do downsamples
            # TODO TODO TODO FIX ALL THIS DO NOT TRUST THIS AT ALL!!!! YOU FIXED UPSAMPLING TO HALVE THE CHANNELS!!!!!!
            in_channels = out_channels * 2 # multiplying by 2 because we'll do channel-wise concatenation with the output from the encoder (i.e. "down") portion of the U-Net 
            out_channels = in_channels // 4 # as we upsample, we halve the number of channels
            block = ResNetBlock(in_channels=in_channels, out_channels=out_channels, d=self.d, dropout=self.dropout)
            self.up_blocks.append(block)
        





    
    def forward(self, x, t, y):
        """
        `x` has shape (B, C, H, W) --- batch of noise samples which are the same shape as a batch of images
        `t` has shape (B, 1) --- batch of times which are scalars in [0, 1]
        `y` has shape (B, 1) --- batch of class labels which are in {0, ..., num_classes}. Class labels in {0, ..., num_classes-1}
        correspond to the regular class labels and a class label of num_classes corresponds to the empty class label, which is used for classifier-free guidance (CFG)

        Remember that this model, when used in a flow model, is the guided vector field u_t^{theta}(x|y) which induces
        the ODE 
            dx/dt = u_t^{theta}(x|y)
        where theta refers to the model's learned parameters and the vector field is "guided" by the class label y.
        """

        B, C, H, W = x.shape
        if H != self.H_init:
            raise ValueError(f"{x.shape=}, {H=}, {self.H_init=}")
        if C != self.channels:
            raise ValueError(f"{x.shape=}, {C=}, {self.channels=}")

        # TODO remember to normalize before attention and to do residual connection with attention
