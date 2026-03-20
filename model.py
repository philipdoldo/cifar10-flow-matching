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
        super().__init__()
        self.d = d
        self.hidden_dim = hidden_dim

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





class Downsample(nn.Module):
    """
    Downsampling layer that halves the spatial dimensions (height and width)

    Input has shape (B, H, W, C) where C is the number of input channels and the output should have shape (B, H/2, W/2, C)
    where H and W are assumed to be even

    Originally, I was going to use nn.MaxPool2d(kernel_size=2, stride=2) which considers a 2x2 grid (starting in the top-left corner
    with no padding, so that the entire 2x2 grid is fully contained within the image) and consolidates ("pools") the 4 values in the
    2x2 grid into a single value by taking the maximum of those 4 values. Note that the stride of 2 applies both horizontally and
    vertically so no two 2x2 grids will overlap and the width and height will be reduced by a factor of 2 (this assumes width and height
    were both even, of course). This pooling process is performed independently for each channel.

    However, instead of doing 2x2 max pooling with a stride of 2, you can do a 3x3 convolution with a stride of 2, a padding of 1, and
    using the same number of filters as there are input channels (so the output has the same number of channels that the input has) and
    this will also reduce the spatial dimensions by a factor of 2 but now instead of arbitrarily restricting the model to do max pooling,
    we are effectively allowing the model to learn what kind of (potentially complicated) pooling to do. Each filter has shape (3,3,C) and
    we are using C filters, so this comes at the cost of adding 9*C^2 parameters, but it might learn a smarter pooling approach. I got this
    idea from https://github.com/KellyYutongHe/cmu-10799-diffusion/blob/main/src/models/blocks.py#L254 where I also took some other minor
    inspirations from here and there.

    `channels` is an integer which is the number of input channels (and output channels)
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # out_channels is the number of filters, an integer for kernel_size gives a square grid, so kernel_size=3 gives a 3x3 grid
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        `x` has shape (B, H, W, C) where B is batch size, H is height, W is width, and C is channels
        output will have shape
        """
        B, H, W, C = x.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"{x.shape=}, height and width should be even but {H=}, {W=}")
        if C != self.channels:
            raise ValueError(f"{x.shape=}, {C=}, {self.channels=}")
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling layer that doubles the spatial dimensions (height and width)

    Input has shape (B, H, W, C) where C is the number of input channels and the output should have shape (B, 2*H, 2*W, C)

    Originally I was going to do a "transposed convolution" which uses C filters of of shape (2, 2, C) and for each pixel value
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
    
    So what does `F.interpolate(x, scale_factor=2, mode='nearest')` actually do? Suppose x has shape (B, H, W, C), consider a single
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
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        B, H, W, C = x.shape
        if C != self.channels:
            raise ValueError(f"{x.shape=}, {C=}, {self.channels=}")
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):

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
