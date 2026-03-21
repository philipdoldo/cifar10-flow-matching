import torch
import torch.nn as nn
import torch.functional as F


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

    Input has shape (B, C, H, W) where C is the number of input channels and the output should have shape (B, C, 2*H, 2*W)

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
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"{x.shape=}, {C=}, {self.channels=}")
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Block(nn.Module):
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


class UNet(nn.Module):
    """
    Inspired by https://arxiv.org/abs/1505.04597 but not intended to be a faithful replication of their implementation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, x, t, y):
        """
        `x` has shape (B, 1, 28, 28) --- batch of noise samples which are the same shape as an MNIST image (batch size B)
        `t` has shape (B, 1) --- batch of times which are scalars in [0, 1]
        `y` has shape (B, 1) --- batch of class labels which are in {0, ..., 10}. Class labels in {0, ..., 9} correspond to MNIST 
        digits and a class label of 10 corresponds to the empty class label, which is used for classifier-free guidance (CFG)

        Remember that this model, when used in a flow model, is the guided vector field u_t^{theta}(x|y) which induces
        the ODE 
            dx/dt = u_t^{theta}(x|y)
        where theta refers to the model's learned parameters and the vector field is "guided" by the class label y.
        """
        pass
