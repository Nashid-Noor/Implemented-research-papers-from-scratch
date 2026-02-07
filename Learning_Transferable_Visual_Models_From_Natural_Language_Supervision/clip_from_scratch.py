import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class QuickGELU(nn.Module):
   
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
 
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
 
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
   
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        
        # Create patch embedding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # Calculate number of patches
        self.num_patches = (input_resolution // patch_size) ** 2
        
        # Initialize parameters with scaled normal distribution
        scale = width ** -0.5
        
        # Class token and positional embedding
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches + 1, width))
        
        # Layer norm before transformer
        self.ln_pre = nn.LayerNorm(width)

        # Transformer backbone
        self.transformer = Transformer(width, layers, heads)

        # Layer norm after transformer
        self.ln_post = nn.LayerNorm(width)
        
        # Projection to output dimension
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        # Input shape: [batch, 3, resolution, resolution]
        x = self.conv1(x)  # Shape: [batch, width, grid, grid]
        
        # Reshape to sequence of patches
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, width, grid*grid]
        x = x.permute(0, 2, 1)  # [batch, grid*grid, width]
        
        # Add class token and positional embedding
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
            x
        ], dim=1)  # [batch, grid*grid+1, width]
        
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        # Apply transformer
        x = x.permute(1, 0, 2)  # [grid*grid+1, batch, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch, grid*grid+1, width]
        
        # Use class token for image representation
        x = self.ln_post(x[:, 0, :])  # [batch, width]
        
        # Project to output dimension
        if self.proj is not None:
            x = x @ self.proj  # [batch, output_dim]
            
        return x


class TextTransformer(nn.Module):

    def __init__(self, context_length=77, vocab_size=49408, width=512, heads=8, layers=12, output_dim=512):
        super().__init__()
        self.context_length = context_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, width)
        
        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        
        # Causal attention mask
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # Zero out the lower diagonal
        self.register_buffer("attn_mask", mask)
        
        # Transformer backbone
        self.transformer = Transformer(width, layers, heads, self.attn_mask)
        
        # Layer norm after transformer
        self.ln_final = nn.LayerNorm(width)
        
        # Projection to output dimension
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        
        # Initialize parameters
        self.initialize_parameters()
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        # Initialize projection matrices
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    
    # Fix for the TextTransformer.forward method in clip_paper_implementation.py

    def forward(self, text):
       
        # text: [batch, context_length]
        x = self.token_embedding(text)  # [batch, context_length, width]
        
        # Add positional embedding
        x = x + self.positional_embedding
        
        # Apply transformer
        x = x.permute(1, 0, 2)  # [context_length, batch, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch, context_length, width]
        
        # Layer norm
        x = self.ln_final(x)
        
        # Take features from the end of text sequence
        # Find first occurrence of padding token (0)
        # Convert boolean mask to int first before using argmax
        padding_mask = (text == 0).type(torch.int64)
        
        # Handle cases where there is no padding (padding_mask is all zeros)
        # If sum of padding_mask is 0 (no padding), use the last token
        has_padding = padding_mask.sum(dim=1) > 0
        
        # Get position of the first padding token or the last token
        seq_lengths = padding_mask.argmax(dim=1)
        
        # For sequences without padding, set position to the last token
        seq_lengths[~has_padding] = text.shape[1] - 1
        
        # Get the embedding at the appropriate position for each sequence
        x = x[torch.arange(x.shape[0]), seq_lengths]
        
        # Project to output dimension
        x = x @ self.text_projection
        
        return x


class CLIP(nn.Module):
    def __init__(self, 
                 input_resolution=224,
                 patch_size=32,
                 vision_width=768,
                 vision_layers=12,
                 vision_heads=12,
                 context_length=77,
                 vocab_size=49408,
                 text_width=512,
                 text_heads=8,
                 text_layers=12,
                 embed_dim=512,
                 temperature=0.07):
        super().__init__()
        
        # Vision transformer
        self.visual = VisionTransformer(
            input_resolution=input_resolution,
            patch_size=patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )
        
        # Text transformer
        self.text = TextTransformer(
            context_length=context_length,
            vocab_size=vocab_size,
            width=text_width,
            heads=text_heads,
            layers=text_layers,
            output_dim=embed_dim
        )
        
        # Logit scale (temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def encode_image(self, image):
        return F.normalize(self.visual(image), dim=-1)
    
    def encode_text(self, text):
        return F.normalize(self.text(text), dim=-1)
    
    def forward(self, image, text):
        # Encode image and text
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Scaled pairwise cosine similarities
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text


def create_clip_model(model_type):
    if model_type == "ViT-B/32":
        return CLIP(
            input_resolution=224,
            patch_size=32,
            vision_width=768,
            vision_layers=12,
            vision_heads=12,
            context_length=77,
            vocab_size=49408,
            text_width=512,
            text_heads=8,
            text_layers=12,
            embed_dim=512
        )
    elif model_type == "ViT-B/16":
        return CLIP(
            input_resolution=224,
            patch_size=16,
            vision_width=768,
            vision_layers=12,
            vision_heads=12,
            context_length=77,
            vocab_size=49408,
            text_width=512,
            text_heads=8,
            text_layers=12,
            embed_dim=512
        )
    elif model_type == "ViT-L/14":
        return CLIP(
            input_resolution=224,
            patch_size=14,
            vision_width=1024,
            vision_layers=24,
            vision_heads=16,
            context_length=77,
            vocab_size=49408,
            text_width=768,
            text_heads=12,
            text_layers=12,
            embed_dim=768
        )
    elif model_type == "ViT-S/32":
        return CLIP(
            input_resolution=224,
            patch_size=32,
            vision_width=384,
            vision_layers=12,
            vision_heads=6,
            context_length=77,
            vocab_size=49408,
            text_width=512,
            text_heads=8,
            text_layers=12,
            embed_dim=512
        )
    elif model_type == "toy":
        return CLIP(
            input_resolution=28,
            patch_size=14,
            vision_width=64,
            vision_layers=3,
            vision_heads=4,
            context_length=32,
            vocab_size=256,
            text_width=64,
            text_heads=4,
            text_layers=2,
            embed_dim=64
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")