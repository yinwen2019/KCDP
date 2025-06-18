import os

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel

EMOTION_LABEL = [
    "A surprise photo of",
    "A happy photo of",
    "A disgust photo of",
    "A fear photo of",
    "A sad photo of",
    "A anger photo of",
]


def cl_loss(image_embeddings, text_embeddings, labels=None, temperature=0.07):
    """
    计算优化后的InfoNCE损失

    参数:
    - image_embeddings: 形状为 [batch_size, embed_dim] 的图像嵌入张量
    - text_embeddings: 形状为 [batch_size, num_texts, embed_dim] 的文本嵌入张量
    - labels: 形状为 [batch_size, 1] 的标签张量，指示每个样本的事实文本索引
    - temperature: 温度参数，用于缩放相似度

    返回:
    - loss: 标量损失值
    """
    batch_size, num_texts, embed_dim = text_embeddings.shape

    # 将图像嵌入扩展为 [batch_size, 1, embed_dim]
    image_embeddings_expanded = image_embeddings.unsqueeze(1)

    # 计算图像嵌入和所有文本嵌入之间的相似度，结果为 [batch_size, num_texts]
    similarities = F.cosine_similarity(image_embeddings_expanded, text_embeddings, dim=2)

    # 应用温度缩放
    similarities /= temperature
    if torch.is_tensor(labels):
        # 使用 labels 创建目标张量
        targets = labels.squeeze()

        # 计算交叉熵损失
        loss = F.cross_entropy(similarities, targets)

        return loss
    else:
        return similarities


class CLIEA(torch.nn.Module):
    def __init__(self):
        super(CLIEA, self).__init__()
        self.emo_labels_emb = torch.load('/home/user/xxx/MultiModal/TGCA_PVT/preprocess/emo_labels_emb.pt')
        self.mul_attn = nn.MultiheadAttention(self.emo_labels_emb.size(-1), 8)
        self.v_align = nn.Linear(in_features=256 * 4, out_features=768)

    def features_align(self, knowledge_emb, v_features, batch_size):
        # T: knowledge_emb [B, 154 ,768]
        t_emb = knowledge_emb.unsqueeze(1).repeat(1, 6, 1, 1)  # [B, 6, L, H]
        emo_emb = self.emo_labels_emb.clone().unsqueeze(0).repeat(batch_size, 1, 1, 1).to(t_emb.device)  # [B, 6, 7, H]
        t_cat = torch.cat((emo_emb, t_emb), dim=2)  # [B, 6, 7+L, H]

        num_sentences = t_cat.size(1)
        seq_len = t_cat.size(2)
        hidden_size = t_cat.size(3)
        # multi-head attention
        t_cat = t_cat.view(batch_size * num_sentences, seq_len, hidden_size).transpose(0, 1)
        attn_output, _ = self.mul_attn(t_cat, t_cat, t_cat)
        t_a = attn_output.mean(dim=0).view(batch_size, num_sentences, hidden_size)  # [B, 6, H]

        # V: features
        v0 = v_features['0']
        v1 = v_features['1']
        v2 = v_features['2']
        v3 = v_features['3']
        v0_pool = torch.mean(v0.view(-1, v0.size(1), v0.size(2) * v0.size(3)), dim=2)
        v1_pool = torch.mean(v1.view(-1, v1.size(1), v1.size(2) * v1.size(3)), dim=2)
        v2_pool = torch.mean(v2.view(-1, v2.size(1), v2.size(2) * v2.size(3)), dim=2)
        v3_pool = torch.mean(v3.view(-1, v3.size(1), v3.size(2) * v3.size(3)), dim=2)
        v_pool = torch.concat([v0_pool, v1_pool, v2_pool, v3_pool], dim=1)
        v_a = self.v_align(v_pool)
        return t_a, v_a

    def forward(self, knowledge_emb, v_features, labels, batch_size):
        t_a, v_a = self.features_align(knowledge_emb, v_features, batch_size)
        loss = cl_loss(v_a, t_a, labels, temperature=0.07)
        return loss

    def get_pseudo_labels(self, knowledge_emb, v_features, labels, batch_size):
        t_a, v_a = self.features_align(knowledge_emb, v_features, batch_size)
        similarities = cl_loss(v_a, t_a, labels, temperature=0.07)
        pseudo_labels = torch.max(similarities, dim=1)
        return pseudo_labels.indices  # [B,1]


if __name__ == "__main__":
    # kalie = CLIEA()
    # knowledge_emb = torch.rand(16, 154, 768)
    # features = {
    #     '0': torch.rand(16, 256, 64, 64),
    #     '1': torch.rand(16, 256, 32, 32),
    #     '2': torch.rand(16, 256, 16, 16),
    #     '3': torch.rand(16, 256, 8, 8),
    # }
    # labels = torch.randint(0, 6, (16, 1))
    # outputs = kalie(knowledge_emb, features, labels, knowledge_emb.size(0))
    # pseudo_labels = kalie.get_pseudo_labels(knowledge_emb, features, None, knowledge_emb.size(0))
    # print(pseudo_labels)
    tokenizer = AutoTokenizer.from_pretrained("/home/user/xxx/PLM/clip-vit-large-patch14")
    clip_text_model = CLIPTextModel.from_pretrained("/home/user/xxx/PLM/clip-vit-large-patch14")
    # init emotion labels embedding
    emo_labels_token = tokenizer(EMOTION_LABEL, padding=True, return_tensors="pt")
    emo_labels_emb = clip_text_model(**emo_labels_token).last_hidden_state
    torch.save(emo_labels_emb, 'emo_labels_emb.pt')
