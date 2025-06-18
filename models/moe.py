import csv

import torch
from torch import nn
import torch.nn.functional as F


class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super(TopKGating, self).__init__()
        # 初始化线性层作为门控机制
        self.gate = nn.Linear(input_dim, num_experts)
        # 设置要选择的顶部专家数量
        self.top_k = top_k

    def forward(self, x):
        # 计算每个专家的分数
        gating_scores = self.gate(x)
        # 选取分数最高的 top_k 个专家，并返回它们的索引和 softmax 权重
        top_k_values, top_k_indices = torch.topk(F.softmax(gating_scores, dim=1), self.top_k)
        return top_k_indices, top_k_values


class Expert(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        # 为每个专家定义一个简单的神经网络
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # 通过专家网络传递输入数据
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts, top_k=2):
        super(MoE, self).__init__()
        # 设置专家数量
        self.num_experts = num_experts
        # 设置类别数量
        self.num_classes = num_classes
        # 初始化 TopK 门控层
        self.gating = TopKGating(input_dim, num_experts, top_k)
        # 创建专家网络的列表，每个专家是一个 Expert 实例
        self.experts = nn.ModuleList([Expert(input_dim, num_classes) for _ in range(num_experts)])

    def forward(self, x):
        # 获取批量大小
        batch_size = x.size(0)

        # 通过门控层获得 top_k 专家的索引和门控权重
        indices, gates = self.gating(x)  # 形状 indices：[batch_size, top_k], gates：[batch_size, top_k]

        # 准备收集选定专家的输出
        expert_outputs = torch.zeros(batch_size, indices.size(1), self.num_classes).to(x.device)

        # 遍历每个样本和其对应的 top_k 专家
        for i in range(batch_size):
            for j in range(indices.size(1)):
                expert_idx = indices[i, j].item()  # 获取专家的索引
                expert_outputs[i, j, :] = self.experts[expert_idx](x[i].unsqueeze(0))

        # 将门控权重扩展到与专家输出相同的维度
        gates = gates.unsqueeze(-1).expand(-1, -1, self.num_classes)  # 形状：[batch_size, top_k, num_classes]

        # 计算加权的专家输出的和
        output = (gates * expert_outputs).sum(1)
        return output, gates.sum(0)  # 返回模型输出和门控使用率以用于负载平衡损失计算


