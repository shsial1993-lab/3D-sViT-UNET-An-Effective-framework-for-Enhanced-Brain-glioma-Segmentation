import torch
import torch.nn as nn
import torch.nn.functional as F
!pip install einops
from einops import rearrange
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class cross_concatenation(nn.Module):
    def __init__(self, in_channels_transformer, in_channels_unet):
        super().__init__()
        # Define convolutional layers for feature transformation
        self.conv_fused = nn.Conv3d(768, 768, kernel_size=1,stride=1)
        self.bn_fused = nn.BatchNorm3d(768)  # Batch normalization for fused features
        self.down_fused = nn.Conv3d(768, 256, kernel_size=1,stride=1)
        self.bn_down = nn.BatchNorm3d(256)  # Batch normalization for fused features
        self.actv=nn.LeakyReLU()

    def forward(self, transformer_output, unet_output):
        # Concatenate along the channel dimension
        fused_feat = torch.cat((transformer_output, unet_output), dim=1)
        fused_feat= self.conv_fused(fused_feat)
        fused_feat = self.bn_fused(fused_feat)
        fused_feat=self.down_fused(fused_feat)
        fused_feat=self.bn_down(fused_feat)
        fused_feat=self.actv(fused_feat)
        return fused_feat



class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 512, 512)) #16x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings[:,:x.size(1)]

class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs
        
class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_fn = self.fn(x)
        return self.norm(x_fn)



class PostNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_dropout = self.dropout(x)
        x_fn = self.fn(x_dropout)
        return self.norm(x_fn)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self,dim,depth,heads,mlp_dim,dropout_rate=0.1,attn_dropout_rate=0.1):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PostNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PostNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)    
class PatchEmbeddings(nn.Module):
    def __init__(self, num_channels=4, patch_dim=16, embedding_dim=512):
        super(PatchEmbeddings, self).__init__()
        self.embedding_convPxP = nn.Conv3d(num_channels, embedding_dim, kernel_size=patch_dim, stride=patch_dim)
    def forward(self, x):
        embeddings = self.embedding_convPxP(x)
        N,C,H,W,D=embeddings.size()
        embeddings=embeddings.view(N,C,H*W*D)#NES
        embeddings=embeddings.permute(0,2,1)#NSE
        return embeddings

class mViT3D(nn.Module):
    def __init__(self, img_dim,patch_dim,num_channels,embedding_dim,num_heads,num_layers,hidden_dim,dropout_rate=0.0,attn_dropout_rate=0.0,):
        super(mViT3D, self).__init__()
        
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers=num_layers
        self.hid_dim=hidden_dim
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.patch_emb=PatchEmbeddings(num_channels=4, patch_dim=16, embedding_dim=512)
        self.position_encoding = LearnedPositionalEncoding(self.seq_length, self.embedding_dim, self.seq_length)
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)
        self.transformer = TransformerModel(embedding_dim,num_layers,num_heads,hidden_dim,self.dropout_rate,self.attn_dropout_rate)
        self.pre_head_ln = nn.LayerNorm(embedding_dim)


    def forward(self, x):
        x=self.patch_emb(x)#NSE
#         print('embedding out shape',x.shape)
        x = self.position_encoding(x)#NSE
        x = self.pe_dropout(x)
        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        x=x.permute(0,2,1)#NES
        x = rearrange(x, 'b c (h w d) -> b c h w d', h=8, w=8, d=8)
        
        return x


class UNetEncoder(nn.Module):
    def __init__(self):
        super(UNetEncoder, self).__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU()
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)
        enc4 = self.encoder4(pool3)
        pool4 = self.pool4(enc4)
        return pool4
    
class UNetBottleneck(nn.Module):
    def __init__(self):
        super(UNetBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.Dropout3d(p=0.5),  # Add dropout after ReLU
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU()
        )

    def forward(self, pool4):
        return self.bottleneck(pool4)



class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        self.upconv4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )
        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Dropout3d(p=0.24)  # Add dropout after ReLU
        )

    def forward(self, pool4, enc4, enc3, enc2, enc1):
        
        upconv4 = self.upconv4(pool4)
        dec4 = self.decoder4(torch.cat([upconv4, enc4], dim=1))

        upconv3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat([upconv3, enc3], dim=1))

        upconv2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([upconv2, enc2], dim=1))

        upconv1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([upconv1, enc1], dim=1))

        return dec1

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        self.conv = nn.Conv3d(in_channels ,out_channels, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)                     
        return x
                                    
class CombinedModel(nn.Module):
    def __init__(self, minivit_config):
        super(CombinedModel, self).__init__()
        self.minivit = mViT3D(**minivit_config)
        self.encoder = UNetEncoder()
        self.bottleneck = UNetBottleneck()
        self.cross_attention = cross_concatenation(512,256)
        self.decoder = UNetDecoder()
        self.fcn=FCNHead(16,4)
        

    def forward(self, x):
        minivit_output = self.minivit(x)
        # Encoder
        enc1 = self.encoder.encoder1(x)
        pool1 = self.encoder.pool1(enc1)
        enc2 = self.encoder.encoder2(pool1)
        pool2 = self.encoder.pool2(enc2)
        enc3 = self.encoder.encoder3(pool2)
        pool3 = self.encoder.pool3(enc3)
        enc4 = self.encoder.encoder4(pool3)
        pool4 = self.encoder.pool4(enc4)
        # Bottleneck
        bottleneck = self.bottleneck.bottleneck(pool4)
        #Cross Attention
        back_important_features = self.cross_attention(minivit_output,bottleneck)
        # Decoder
        upconv4 = self.decoder.upconv4(back_important_features)
        dec4 = self.decoder.decoder4(torch.cat([upconv4, enc4], dim=1))
        upconv3 = self.decoder.upconv3(dec4)
        dec3 = self.decoder.decoder3(torch.cat([upconv3, enc3], dim=1))
        upconv2 = self.decoder.upconv2(dec3)
        dec2 = self.decoder.decoder2(torch.cat([upconv2, enc2], dim=1))
        upconv1 = self.decoder.upconv1(dec2)
        dec1 = self.decoder.decoder1(torch.cat([upconv1, enc1], dim=1))
        output = self.fcn(dec1) #classifier head
        return output

minivit_config = {
    'img_dim': 128,  # Assuming input image dimension is 128x128
    'patch_dim': 16,  # Assuming patch dimension is 8x8
    'num_channels': 4,  # Assuming input has 4 channels
    'embedding_dim': 512,  # Dimensionality of token embeddings
    'num_heads': 8,  # Number of attention heads
    'num_layers': 4,  # Number of layers in the transformer
    'hidden_dim': 4095,  # Dimensionality of the feedforward network hidden layer
    'dropout_rate': 0.5,  # Dropout rate
    'attn_dropout_rate': 0.5,  # Dropout rate for attention layers
}

model = CombinedModel(minivit_config).to(device)
# print(model)