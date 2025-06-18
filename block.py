import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.fuse_num_heads
        self.attention_head_size = int(config.hidden_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_dim, self.all_head_size)
        self.key = Linear(config.hidden_dim, self.all_head_size)
        self.value = Linear(config.hidden_dim, self.all_head_size)

        self.out = Linear(config.hidden_dim, config.hidden_dim)
        self.attn_dropout = Dropout(config.fuse_attention_dropout_rate)
        self.proj_dropout = Dropout(config.fuse_attention_dropout_rate)
        self.conv = nn.Conv1d(config.hidden_dim,config.hidden_dim,kernel_size=1)
        self.norm = LayerNorm(config.hidden_dim)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def context_aware(self, x):
        x = x + self.conv(x.permute(0,2,1)).permute(0,2,1)  # v2
        #x = x + cbam(x.permute(0,2,1)).permute(0,2,1)
        return torch.mean(x, dim=0) # .values

    def context(self, features_tensor, id_tensor):
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
        encoded_contexts = [self.context_aware(torch.stack(encoded_contexts_dict[current_id]))
                            for current_id in sorted(encoded_contexts_dict.keys())]
        contexts = []
        for i in range(batch_size):
            contexts.append(encoded_contexts[id_tensor[i]-1].unsqueeze(0))
        contexts = torch.cat(contexts, dim=0)
        contexts = contexts + features_tensor
        return contexts 


    def forward(self, hidden_states, tid):
        topic_states = self.context(hidden_states, tid)

        mixed_query_layer = self.query(hidden_states)
        topic_states = self.query(topic_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)  # Multi-head:12*64=768
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        topic_layer = self.transpose_for_scores(topic_states)
        
        ### v2 改了这里 

        attention_scores = torch.matmul(topic_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)
        v_t =  torch.matmul(attention_probs, value_layer)

        attention_scores = torch.matmul(query_layer, topic_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v_t)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output #, weights



class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_dim, config.fuse_mlp_dim)
        self.fc2 = Linear(config.fuse_mlp_dim, config.hidden_dim)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.fuse_dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class FuseBlock(nn.Module):
    def __init__(self, config):
        super(FuseBlock, self).__init__()
        self.hidden_size = config.hidden_dim
        self.attention_norm = LayerNorm(config.hidden_dim, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_dim, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x, tid):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x, tid)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x[:,0]

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)



    
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv1d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, N = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, N
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, N
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, N
        x2 = Rearrange('b c t n -> b (c t) n')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_gate = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_gate(x)
        x = x * self.spatial_gate(x)

        return x
