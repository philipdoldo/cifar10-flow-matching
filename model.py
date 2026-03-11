import torch
import torch.nn as nn



class SinusoidalEmbedding(nn.Module):

    def __init__(self, d, base):
        super().__init__()
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

class UNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, x, t, y):
        """
        `x` has shape (B, 1, 28, 28) --- batch of noise samples which are the same shape as an MNIST image (batch size B)
        `t` has shape (B, 1) --- batch of times which are scalars in [0, 1]
        `y` has shape (B, 1) --- baych of class labels which are in {0, ..., 10}. Class labels in {0, ..., 9} correspond to MNIST 
        digits and a class label of 10 corresponds to the empty class label, which is used for classifier-free guidance (CFG)

        Remember that this model, when used in a flow model, is the guided vector field u_t^{theta}(x|y) which induces
        the ODE 
            dx/dt = u_t^{theta}(x|y)
        where theta refers to the model's learned parameters and the vector field is "guided" by the class label y.
        """
        pass
