import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce



class PoseEmbedding(nn.Module):
    def __init__(self,
        num_patches: int = 30, 
        in_features: int = 26, 
        emb_size   : int = 768,
        device="cpu"
    ):
        
        super(PoseEmbedding, self).__init__()
        self.in_features = in_features
        self.device = device
        self.projection = nn.Sequential(
            nn.Linear(in_features, emb_size)
        )

        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        # position embedding
        self.positions = nn.Parameter(torch.randn(num_patches + 1, emb_size))

             
    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        batch_size, _, _ = x.shape    
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size).to(self.device) # TODO
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x
    


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0, device="cpu"):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.device = device

        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        x = x.to(self.device)
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    

class ResidualAdd(nn.Module):
    """
    wrapper to perform the residual addition
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 device = "cpu",
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, device=device,**kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )



class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, device = "cpu",**kwargs):
        super().__init__(
            *[TransformerEncoderBlock(**kwargs) for _ in range(depth)]
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 20):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes)
        )


    
class ViT(nn.Sequential):
    def __init__(self,     
                num_poses: int = 30, 
                in_features: int = 26, 
                emb_size: int = 768,
                depth: int = 12,
                num_classes: int = 20,
                device="cpu",
                **kwargs):
        super().__init__(
            PoseEmbedding(num_poses, in_features, emb_size,device, device = device),
            TransformerEncoder(depth, emb_size=emb_size,device = device **kwargs),
            ClassificationHead(emb_size, num_classes)
        )


if __name__ == "__main__":
    x = torch.randn(1, 30, 26)
    model = ViT()
    print(model(x).shape)