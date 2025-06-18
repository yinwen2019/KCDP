# Code based on the Pyramid Vision Transformer
# https://github.com/whai362/PVT
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from block import CBAM
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = [
    'pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large'
]

class se_block(nn.Module):
    def __init__(self,channels,ratio=16):
        super(se_block, self).__init__()
        # 空间信息进行压缩
        self.avgpool=nn.AdaptiveAvgPool2d(1)
 
        # 经过两次全连接层，学习不同通道的重要性
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_ = x.size() #取出batch size和通道数
 
        # b,c,w,h->b,c,1,1->b,c 压缩与通道信息学习
        avg = self.avgpool(x).view(b,c)
 
        #b,c->b,c->b,c,1,1 激励操作
        y = self.fc(avg).view(b,c,1,1) 
        return x * y.expand_as(x)


class FFT_Attention(nn.Module):
    def __init__(self, in_channels=3, norm='ortho'):
        super().__init__()
        # self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2).float() * 0.01)
        self.SE1 = se_block(channels=in_channels)
        self.SE2 = se_block(channels=in_channels)
        self.norm = norm
        self.r = nn.Parameter(torch.zeros(in_channels, in_channels))
        self.i = nn.Parameter(torch.zeros(in_channels, in_channels))
        self.rb = nn.Parameter(torch.zeros(in_channels))
        self.ib = nn.Parameter(torch.zeros(in_channels))
        trunc_normal_(self.r, std=.02)
        trunc_normal_(self.i, std=.02)
        trunc_normal_(self.rb, std=.02)
        trunc_normal_(self.ib, std=.02)

    def forward(self, inputs):
       
        B, C, H, W = inputs.shape
        
        y = torch.fft.rfft2(inputs, norm=self.norm)

        y_real = y.real  # 实部
        y_imag = y.imag  # 虚部
        y_real = y_real.permute(0,2,3,1)
        y_imag = y_imag.permute(0,2,3,1)
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', y_real, self.r) - \
            torch.einsum('bijd,dd->bijd', y_imag, self.i) + \
            self.rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', y_imag, self.r) + \
            torch.einsum('bijd,dd->bijd', y_real, self.i) + \
            self.ib
        )

        y_real = y_real.permute(0,3,1,2)
        y_imag = y_imag.permute(0,3,1,2)
        y_real = y_real + self.SE1(o1_real.permute(0,3,1,2))
        y_imag = y_imag + self.SE2(o1_imag.permute(0,3,1,2))

        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Local_ReAttention(nn.Module):
    def __init__(self, alpha):
        super(Local_ReAttention, self).__init__()
        self.alpha = alpha

    def forward(self, x, last=False):
        '''
        x[i]: Batchsize * num_heads_l * (H_l*W_l) * (H_l*W_l/sr_ratio_l^2)
        '''
        length = len(x)
        last_map = x[0]
        _,_,h,w = last_map.shape
        ratio = h//(w-1) if h!=w else 1
        for i in range(1, length):
            if not last:
                if i==1:
                    last_map = torch.matmul(x[i], last_map.permute(0,1,3,2))
                else:
                    if i%2==0:
                        last_map = torch.matmul(x[i].permute(0,1,3,2), last_map)
                    else:
                        last_map = torch.matmul(x[i], last_map)
            else:
                last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:]

        # _, max_inx = last_map.max(2)  # Find max index in channel-dim, [bs, 12]
        _, topk_inx = last_map.topk(self.alpha*ratio, dim=2)
        return topk_inx

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, locals=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.locals = locals

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            if self.locals:
                head = x[:,0,:]
                x_ = x[:,1:,:].permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = torch.cat([head.unsqueeze(1), x_], dim=1)
            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn_prob = self.softmax((q @ k.transpose(-2, -1)) * self.scale)
        # L R-A Module
        attn = self.attn_drop(attn_prob)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_prob

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, locals=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, locals=locals)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        xx, weight = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(xx)
        # x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x , weight


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.fft = FFT_Attention(in_chans)


    def forward(self, x):
        B, C, H, W = x.shape
        x = self.fft(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, locals=[0, 0, 0, 0], alpha=8):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims
        self.locals = locals

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i], locals=locals[i])
                for j in range(depths[i])])
            cur += depths[i]
            # 2024.03.24 22：30
            conv = nn.Conv1d(embed_dims[i], embed_dims[i], kernel_size=1)
            #cbam = nn.Sequential(
                #nn.Conv1d(embed_dims[i], embed_dims[i], kernel_size=1),
            #    CBAM(embed_dims[i]),
            #)
            ## LRA module
            if locals[i]==1:
                cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[i]))
                local_proj = nn.Sequential(
                                            nn.Linear(embed_dims[i], embed_dims[i]),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims[i], embed_dims[3])
                                            )
                global_proj = nn.Sequential(
                                            nn.Linear(embed_dims[i], embed_dims[i]),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims[i], embed_dims[3])
                                            )
                

                setattr(self, f"cls_token{i + 1}", cls_token)
                setattr(self, f"local_proj{i + 1}", local_proj)
                setattr(self, f"global_proj{i + 1}", global_proj)

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)
            # setattr(self, f"cbam{i + 1}", cbam)
            setattr(self, f"conv{i + 1}", conv)
        self.norm = norm_layer(embed_dims[3])
        self.softmax = nn.Softmax(dim=-1)
        # Local Re-Attention Module
        self.LRA = Local_ReAttention(alpha)

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
            if locals[i]==1:
                cls_token = getattr(self, f"cls_token{i + 1}")
                trunc_normal_(cls_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    
    def context_aware(self, x, conv):
        x = x + conv(x.permute(0,2,1)).permute(0,2,1) 
        #x = x + cbam(x.permute(0,2,1)).permute(0,2,1)
        return torch.mean(x, dim=0) # .values

    def context(self, features_tensor, id_tensor, conv):
    # 获取特征张量的维度信息
        batch_size,  num_patches, feature_dim = features_tensor.size()
        
        # 初始化一个字典来存储编码结果，键是ID，值是编码后的上下文特征
        encoded_contexts_dict = {}

        # 遍历每个样本
        for i in range(batch_size):
            current_id = id_tensor[i].item()
            
            # 如果当前ID尚未在字典中，则添加一个键值对
            if current_id not in encoded_contexts_dict:
                encoded_contexts_dict[current_id] = []

            # 将当前样本的特征添加到对应ID的列表中
            encoded_contexts_dict[current_id].append(features_tensor[i])

        # 对字典中的每个值（列表）计算平均值，得到编码后的上下文特征
        encoded_contexts = [self.context_aware(torch.stack(encoded_contexts_dict[current_id]), conv)
                            for current_id in sorted(encoded_contexts_dict.keys())]
        contexts = []
        for i in range(batch_size):
            contexts.append(encoded_contexts[id_tensor[i]-1].unsqueeze(0))
        contexts = torch.cat(contexts, dim=0)
        contexts = contexts + features_tensor
        return contexts 

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        # if H * W == self.patch_embed1.num_patches:
        if H * W == patch_embed.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x, tid):
        B = x.shape[0]
        global_tokens = []
        local_tokens = []
        topic_tokens = []

        for i in range(self.num_stages):
            attn_weights = []
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            # cbam = getattr(self, f"cbam{i + 1}")
            conv = getattr(self, f"conv{i + 1}")
            if self.locals[i]==1:
                cls_token = getattr(self, f"cls_token{i + 1}")
                global_proj = getattr(self, f"global_proj{i + 1}")
                local_proj = getattr(self, f"local_proj{i + 1}")
                
            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            elif self.locals[i]==1:
                cls_token = cls_token.expand(B, -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed, patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)

            for blk in block:
                x, weight = blk(x, H, W)
                attn_weights.append(weight)

            x_c = self.context(x, tid, conv)

            ## Local Re-Attention
            if i == self.num_stages-1 or self.locals[i]==1:
                local_inx = self.LRA(attn_weights, i==self.num_stages-1)
                local_inx = local_inx + 1
                local_inx = local_inx.view(B,-1)
                parts = []
                for j in range(B):
                    local_attn = x[j, local_inx[j,:]]
                    # local_mean = torch.mean(x[j], dim=0, keepdim=True)
                    # local_mean = local_mean.expand_as(local_attn)
                    # local_attn = local_attn + self.softmax(local_attn @ local_mean.transpose(-2,-1)) @ local_mean
                    parts.append(local_attn)
                parts = torch.stack(parts)
                parts_max = torch.max(parts, dim=1, keepdim=True).values
                parts_max = parts_max.expand_as(parts)
                parts = parts + self.softmax( parts @ parts_max.transpose(-2,-1)) @ parts_max
                if self.locals[i]==1:
                    parts = local_proj(parts)
                #else:
                #    parts = torch.stack(parts)
                local_tokens.append(parts)
            ## Global
            if i != self.num_stages - 1:
                if self.locals[i]==1:
                    global_tokens.append(global_proj(x[:,0,:]))
                    topic_tokens.append(global_proj(x_c[:,0,:]))
                    x = x[:,1:,:]
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            else:
                global_tokens.append(x[:,0,:])
                topic_tokens.append(x_c[:,0,:])
    

        global_tokens = torch.stack(global_tokens, dim=1).flip(dims=[1])
        topic_tokens = torch.stack(topic_tokens, dim=1).flip(dims=[1])
        global_tokens = global_tokens + self.softmax(global_tokens @ topic_tokens.transpose(-2,-1)) @ topic_tokens
        local_tokens = torch.cat(local_tokens, dim=1).flip(dims=[1])
        x = torch.cat([global_tokens, local_tokens], dim=1)  # Global+Local
        x = self.norm(x)

        return x

    def forward(self, x, tid):
        x = self.forward_features(x, tid)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pvt_tiny(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_small(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_large(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_huge_v2(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[128, 256, 512, 768], num_heads=[2, 4, 8, 12], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 10, 60, 3], sr_ratios=[8, 4, 2, 1],
        # drop_rate=0.0, drop_path_rate=0.02)
        **kwargs)
    model.default_cfg = _cfg()

    return model

if __name__=='__main__':
    model = PyramidVisionTransformer(
        num_classes=7,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], 
        alpha=8, 
        locals=[0, 0, 0, 0])
    
    ## Load checkpoint
    checkpoint = torch.load('weights/pvt_small.pth', map_location='cpu')
    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model:
            del checkpoint_model[k]
    model.load_state_dict(checkpoint_model, strict=False)
    
    x = torch.randn([1, 3, 448, 448])
    x = x.cuda()
    model = model.cuda()
    y = model(x)
    print(y.shape)
    # alpha=8, locals:
    # [1, 1, 1, 0]: [1, 996, 512]
    # [0, 1, 1, 0]: [1, 483, 512]
    # [0, 0, 1, 0]: [1, 226, 512]
    # [0, 0, 0, 0]: [1,  65, 512]
