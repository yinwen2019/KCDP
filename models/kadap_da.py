import csv
import json
import os
import pickle
import warnings
from typing import Dict, List, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional
from sklearn import metrics
import wandb
from PIL import Image

from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from timm.data import create_transform
from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ldm.util import instantiate_from_config
from models.cliea import CLIEA
from models.moe import MoE
from models.vpd import UNetWrapper


class EmotionHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.num_classes = 6
        self.in_channels = in_channels
        #self.fc1 = nn.Linear(in_features=in_channels * 4, out_features=self.num_classes)
        self.fc1 = nn.Linear(in_features=in_channels * 4, out_features=self.num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, feature):
        feature0 = feature['0']
        feature1 = feature['1']
        feature2 = feature['2']
        feature3 = feature['3']
        feature0_pool = torch.mean(feature0.view(-1, feature0.size(1), feature0.size(2) * feature0.size(3)), dim=2)
        feature1_pool = torch.mean(feature1.view(-1, feature1.size(1), feature1.size(2) * feature1.size(3)), dim=2)
        feature2_pool = torch.mean(feature2.view(-1, feature2.size(1), feature2.size(2) * feature2.size(3)), dim=2)
        feature3_pool = torch.mean(feature3.view(-1, feature3.size(1), feature3.size(2) * feature3.size(3)), dim=2)
        feature_cat = torch.cat([feature0_pool, feature1_pool, feature2_pool, feature3_pool], dim=1)
        #feature_cat = feature0_pool+feature1_pool+feature2_pool+feature3_pool
        return self.dropout(self.fc1(feature_cat))


class EmotionHeadMoe(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.num_classes = 6
        self.in_channels = in_channels
        self.moe = MoE(input_dim=in_channels * 4, num_classes=6, num_experts=4)

    def forward(self, feature, c_feature, t_feature):
        feature0 = feature['0']
        feature1 = feature['1']
        feature2 = feature['2']
        feature3 = feature['3']
        feature0_pool = torch.mean(feature0.view(-1, feature0.size(1), feature0.size(2) * feature0.size(3)), dim=2)
        feature1_pool = torch.mean(feature1.view(-1, feature1.size(1), feature1.size(2) * feature1.size(3)), dim=2)
        feature2_pool = torch.mean(feature2.view(-1, feature2.size(1), feature2.size(2) * feature2.size(3)), dim=2)
        feature3_pool = torch.mean(feature3.view(-1, feature3.size(1), feature3.size(2) * feature3.size(3)), dim=2)
        feature_cat = torch.cat([feature0_pool, feature1_pool, feature2_pool, feature3_pool], dim=1)
        output, gating_weights = self.moe(feature_cat)
        return output, gating_weights


def moe_loss(output, target, gating_weights, lambda_balance=0.1):
    # 标准损失（例如交叉熵损失）
    # output 是模型的输出，target 是真实的标签
    standard_loss = F.cross_entropy(output, target)

    # 负载平衡损失
    # gating_weights 是门控权重，表示每个专家的使用率
    # 使用标准差来衡量各专家使用率的平衡程度
    balance_loss = torch.std(gating_weights)

    # 总损失
    # 结合标准损失和负载平衡损失，lambda_balance 是一个超参数，用于控制负载平衡损失在总损失中的比重
    total_loss = standard_loss + lambda_balance * balance_loss
    return total_loss


class KADAPDA(pl.LightningModule):

    def __init__(self, cfg=None,
                 train_dataset=None,
                 test_cfg=None,
                 unet_config=dict(),
                 use_decoder=False,
                 freeze_backbone=False,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        # get config
        self.transform = None
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.test_cfg = test_cfg

        self.freeze_backbone = freeze_backbone

        # turn text conditioning into list
        if '+' in self.cfg['text_conditioning']:
            self.cfg['text_conditioning'] = self.cfg['text_conditioning'].split('+')
        else:
            self.cfg['text_conditioning'] = [self.cfg['text_conditioning']]

        # check sd ckpt if model is there if not download
        sd_ckpt = "v1-5-pruned-emaonly.ckpt"
        sd_dir = "weights/sd"
        sd_path = sd_dir + '/' + sd_ckpt
        if not os.path.exists(sd_dir):
            os.makedirs(sd_dir)
        if not os.path.exists(os.path.join(sd_dir, sd_ckpt)):
            print('start downloading stable-diffusion from huggingface')
            hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename=sd_ckpt, local_dir=sd_dir)

        # load sd config
        config = OmegaConf.load('stable_diffusion/configs/stable-diffusion/v1-inference.yaml')
        config.model.params.ckpt_path = f'./{sd_path}'
        # instantiate sd model
        sd_model = instantiate_from_config(config.model)

        # handle logic for using scaled encoder
        if not self.cfg.get('use_scaled_encode', False):
            self.encoder_vq = sd_model.first_stage_model
            sd_model.first_stage_model = None
            if not use_decoder:
                del self.encoder_vq.decoder
            ### set grads to zero to be safe
            for param in self.encoder_vq.parameters():
                param.requires_grad = False
        else:
            if not use_decoder:
                del sd_model.first_stage_model.decoder

        # instantiate unet wrapper
        self.model = UNetWrapper(sd_model.model, **unet_config)
        sd_model.model = None
        if 'blip' in self.cfg["text_conditioning"]:
            for param in sd_model.cond_stage_model.parameters():
                param.requires_grad = False
        self.preprocess_path = cfg['preprocess_path']
        self.train_triples_file = None
        if self.train_dataset == 'Emoset':
            self.train_triples_file_s = os.path.join(self.preprocess_path, 'triples/Emoset_train_triples.json')
            self.train_triples_file_t = os.path.join(self.preprocess_path, 'triples/SERCaption_train_triples.json')
        elif self.train_dataset == 'SER':
            self.train_triples_file_s = os.path.join(self.preprocess_path, 'triples/SERCaption_train_triples.json')
            self.train_triples_file_t = os.path.join(self.preprocess_path, 'triples/Emoset_train_triples.json')
        self.val_triples_file0 = os.path.join(self.preprocess_path, 'triples/Emoset_val_triples.json')
        self.val_triples_file1 = os.path.join(self.preprocess_path, 'triples/SERCaption_val_triples.json')

        self.use_decoder = use_decoder
        self.sd_model = sd_model

        self.with_neck = True

        enc_mid_channels, enc_end_channels = self.compute_decoder_head_shapes()
        self.fpn = FeaturePyramidNetwork(in_channels_list=[320, enc_mid_channels, enc_end_channels, 1280],
                                         out_channels=256,
                                         extra_blocks=None,
                                         norm_layer=None)
        self.cls_head = None
        if self.with_neck:
            self.cls_head = EmotionHeadMoe(in_channels=256)
        else:
            self.cls_head = EmotionHead(in_channels=256)

        self.ce_loss = nn.CrossEntropyLoss()

        self.kalie = CLIEA()
        self.test_all_preds = []
        self.test_all_targets = []

        self.val_emo_all_preds = []
        self.val_ser_all_preds = []
        self.val_emo_all_targets = []
        self.val_ser_all_targets = []

    def compute_decoder_head_shapes(self):
        text_cond = self.cfg['text_conditioning']
        enc_mid_channels = 640
        enc_end_channels = 1280
        if self.cfg['append_self_attention']:
            enc_mid_channels += 1024
            enc_end_channels += 256

        if 'blip' in text_cond:
            enc_mid_channels += 77
            enc_end_channels += 77

        if 'triple' in text_cond:
            enc_mid_channels += 77
            enc_end_channels += 77
        return enc_mid_channels, enc_end_channels

    def create_text_embeddings(self, captions=None, triples_tensors=None):
        text_cond = self.cfg['text_conditioning']
        conds = []
        _cs = []
        if 'blip' in text_cond:
            _cs = [self.sd_model.get_learned_conditioning([caption]) for caption in captions]
            c = torch.cat(_cs, dim=0)  # [B, H]
            conds.append(c)
        if 'triple' in text_cond:
            conds.append(triples_tensors)
        c_crossattn = torch.cat(conds, dim=1)  # [B, H1+H2]
        return c_crossattn

    def extract_feat(self, orig_images_tensors, captions, triples_tensors):
        """Extract features from images."""
        if self.cfg.get('use_scaled_encode', False):
            with torch.no_grad():
                latents = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(orig_images_tensors))
        else:
            with torch.no_grad():
                latents = self.encoder_vq.encode(orig_images_tensors)
            latents = latents.mode().detach()

        c_crossattn = self.create_text_embeddings(captions=captions, triples_tensors=triples_tensors)

        t = torch.from_numpy(np.array([1])).to(orig_images_tensors.device)
        outs = self.model(latents, t, c_crossattn=[c_crossattn])
        return c_crossattn, outs

    def forward(self, x, y, idxs, captions, triples_tensors):
        orig_images = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])(x) for x in x]
        orig_images_tensors = torch.stack(orig_images).to(self.device)

        knowledge_emb, features = self.extract_feat(orig_images_tensors, captions, triples_tensors)
        source_loss = 0
        # supervise learning
        feat_names = ['0', '1', '2', '3']
        features = dict(zip(feat_names, features))
        features = self.fpn(features)

        if self.with_neck:
            y_pred, w = self.cls_head(features, triples_tensors, None)
            source_loss += moe_loss(y_pred, y, w)
        else:
            y_pred = self.cls_head(features)
        source_loss += self.ce_loss(y_pred, y)
        # Language-Image Emotional alignment learning
        # input: knowledge_emb, features
        # output: contrastive loss from language-image alignment of counterfactual label words
        source_loss += self.kalie(knowledge_emb, features, y, knowledge_emb.size(0))
        return source_loss, y_pred

    def forward_da(self, x, captions, triples_tensors):
        orig_images = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])(x) for x in x]
        orig_images_tensors = torch.stack(orig_images).to(self.device)
        knowledge_emb, features = self.extract_feat(orig_images_tensors, captions, triples_tensors)

        feat_names = ['0', '1', '2', '3']
        features = dict(zip(feat_names, features))
        features = self.fpn(features)
        total_loss = 0
        y_pseudo = self.kalie.get_pseudo_labels(knowledge_emb, features, None, knowledge_emb.size(0))
        if self.with_neck:
            y_pred, w = self.cls_head(features, triples_tensors, None)
            total_loss += moe_loss(y_pred, y_pseudo, w)
        else:
            y_pred = self.cls_head(features)
            total_loss += self.ce_loss(y_pred, y_pseudo)

        return total_loss, y_pred

    def inference(self, x, idxs, captions):
        orig_images = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])(x) for x in x]
        orig_images_tensors = torch.stack(orig_images).to(self.device)
        features = self.extract_feat(orig_images_tensors, idxs, captions)

        feat_names = ['0', '1', '2', '3']
        features = dict(zip(feat_names, features))

        features = self.fpn(features)
        y_pred = self.cls_head(features=features)

        return y_pred

    def configure_optimizers(self):
        # TODO: double check here
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # have differernt learning rate for different layers
        # parameters to optimize
        lesslr_no_decay = list()
        lesslr_decay = list()
        no_lr = list()
        no_decay = list()
        decay = list()
        for name, m in self.named_parameters():
            if 'unet' in name and 'norm' in name:
                lesslr_no_decay.append(m)
            elif 'unet' in name:
                lesslr_decay.append(m)
            elif 'encoder_vq' in name:
                no_lr.append(m)
            elif 'norm' in name:
                no_decay.append(m)
            else:
                decay.append(m)

        if self.freeze_backbone:
            params_to_optimize = [
                {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale': 0.0},
                {'params': lesslr_decay, 'lr_scale': 0.0},
                {'params': no_lr, 'lr_scale': 0.0},
                {'params': no_decay, 'weight_decay': 0.0},
            ]
        else:
            params_to_optimize = [
                {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale': 0.01},
                {'params': lesslr_decay, 'lr_scale': 0.01},
                {'params': no_lr, 'lr_scale': 0.0},
                {'params': no_decay, 'weight_decay': 0.0},
                {'params': decay}
            ]
        optimizer = torch.optim.AdamW(params_to_optimize,
                                      lr=0.00001,
                                      # lr=0.000005,
                                      weight_decay=1e-2,
                                      amsgrad=False
                                      )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (1 - x / (
                self.cfg["dataset_len"] * self.cfg["max_epochs"])) ** 0.9)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}]

    def training_step(self, batch, batch_idx):
        total_loss = 0
        # TODO：Training source data
        img_paths, targets, idxs, captions = batch[0]
        imgs = []
        for path in img_paths:
            img_meta = Image.open(path).convert("RGB")
            self.transform = build_transform_da(mode='train')
            img = self.transform(img_meta)
            imgs.append(img)
            # load triples tensors
        triples_tensors = []
        with open(self.train_triples_file_s, 'r') as f:
            image_triples = json.load(f)
            for i, idx in enumerate(idxs):
                i_triples = image_triples[idx]
                i_triples_tensors = [
                    self.sd_model.get_learned_conditioning([triple['subject'] + triple['verb'] + triple['object']]) for
                    triple in i_triples]
                if len(i_triples_tensors) == 0:
                    triples_tensors.append(self.sd_model.get_learned_conditioning([captions[i]]).squeeze(0))
                else:
                    triples_tensors.append(torch.sum(torch.stack(i_triples_tensors), dim=0).squeeze(0))
        triples_tensors = torch.stack(triples_tensors).to(self.device)
        loss, y_pred = self(imgs, targets, idxs, captions, triples_tensors)
        total_loss += loss
        self.log("train_s_loss", loss, batch_size=len(batch))
        del img_paths
        del targets
        del idxs
        del captions
        del triples_tensors
        del imgs
        del loss
        # TODO: Unsupervised training target data
        img_paths, targets, idxs, captions = batch[1]
        imgs = []
        for path in img_paths:
            img_meta = Image.open(path).convert("RGB")
            self.transform = build_transform_da(mode='train')
            img = self.transform(img_meta)
            imgs.append(img)
        # load triples tensors
        triples_tensors = []
        with open(self.train_triples_file_t, 'r') as f:
            image_triples = json.load(f)
            for i, idx in enumerate(idxs):
                i_triples = image_triples[idx]
                i_triples_tensors = [
                    self.sd_model.get_learned_conditioning([triple['subject'] + triple['verb'] + triple['object']])
                    for
                    triple in i_triples]
                if len(i_triples_tensors) == 0:
                    triples_tensors.append(self.sd_model.get_learned_conditioning([captions[i]]).squeeze(0))
                else:
                    triples_tensors.append(torch.sum(torch.stack(i_triples_tensors), dim=0).squeeze(0))
        triples_tensors = torch.stack(triples_tensors).to(self.device)
        loss, y_pred = self.forward_da(imgs, captions, triples_tensors)
        total_loss += loss
        self.log("train_t_loss", loss, batch_size=len(batch))
        del img_paths
        del targets
        del idxs
        del captions
        del triples_tensors
        del imgs
        del loss
        #torch.cuda.empty_cache()
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0, plotting=False):
        img_paths, targets, idxs, captions = batch
        imgs = []
        for path in img_paths:
            img_meta = Image.open(path).convert("RGB")
            self.transform = build_transform_da(mode='val')
            img = self.transform(img_meta)
            imgs.append(img)
        # load triples tensors
        triples_tensors = []
        val_triples_file = self.val_triples_file0 if dataloader_idx == 0 else self.val_triples_file1
        with open(val_triples_file, 'r') as f:
            image_triples = json.load(f)
            for i, idx in enumerate(idxs):
                i_triples = image_triples[idx]
                i_triples_tensors = [
                    self.sd_model.get_learned_conditioning([triple['subject'] + triple['verb'] + triple['object']]) for
                    triple in i_triples]
                if len(i_triples_tensors) == 0:
                    triples_tensors.append(self.sd_model.get_learned_conditioning([captions[i]]).squeeze(0))
                else:
                    triples_tensors.append(torch.sum(torch.stack(i_triples_tensors), dim=0).squeeze(0))
        triples_tensors = torch.stack(triples_tensors).to(self.device)

        loss, y_pred = self(imgs, targets, idxs, captions, triples_tensors)
        self.log(f"val_{dataloader_idx}_loss", loss, batch_size=len(batch))

        for y_p in torch.argmax(y_pred, dim=-1):
            if dataloader_idx == 0:
                self.val_emo_all_preds.append(y_p)
            elif dataloader_idx == 1:
                self.val_ser_all_preds.append(y_p)
        for y_ in targets:
            if dataloader_idx == 0:
                self.val_emo_all_targets.append(y_)
            elif dataloader_idx == 1:
                self.val_ser_all_targets.append(y_)

        del img_paths
        del targets
        del idxs
        del captions
        del imgs
        del y_pred
        #torch.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0, plotting=False):
        img_paths, targets, idxs, captions = batch
        imgs = []
        for path in img_paths:
            img_meta = Image.open(path).convert("RGB")
            self.transform = build_transform_da(mode='val')
            img = self.transform(img_meta)
            imgs.append(img)

        with open('/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_CLIEA_ser_l_1k.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            # 将隐藏层向量转为 numpy 并逐行写入
            labels = targets.detach().cpu().numpy()  # [16, 1]
            for label in labels:
                writer.writerow([label])
        # load triples tensors
        triples_tensors = []
        val_triples_file = os.path.join(self.preprocess_path, 'triples/SERCaption_test_triples.json')
        with open(val_triples_file, 'r') as f:
            image_triples = json.load(f)
            for i, idx in enumerate(idxs):
                i_triples = image_triples[idx]
                i_triples_tensors = [
                    self.sd_model.get_learned_conditioning([triple['subject'] + triple['verb'] + triple['object']]) for
                    triple in i_triples]
                if len(i_triples_tensors) == 0:
                    triples_tensors.append(self.sd_model.get_learned_conditioning([captions[i]]).squeeze(0))
                else:
                    triples_tensors.append(torch.sum(torch.stack(i_triples_tensors), dim=0).squeeze(0))
        triples_tensors = torch.stack(triples_tensors).to(self.device)

        loss, y_pred = self(imgs, targets, idxs, captions, triples_tensors)
        with open('/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_CLIEA_ser_v_1k.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            hidden_output_np = y_pred.detach().cpu().numpy()  # [16, 6]
            writer.writerows(hidden_output_np)
        for y_p in torch.argmax(y_pred, dim=-1):
            self.test_all_preds.append(y_p)
        for y_ in targets:
            self.test_all_targets.append(y_)

        del img_paths
        del targets
        del idxs
        del captions
        del imgs
        del y_pred
        # torch.cuda.empty_cache()
        return loss

    def on_validation_epoch_end(self):
        pred = torch.stack(self.val_emo_all_preds).cpu()
        labels = torch.stack(self.val_emo_all_targets).cpu()
        labels_report = metrics.classification_report(labels, pred, digits=4, output_dict=True)
        acc = metrics.accuracy_score(labels, pred)
        print(f"Emoset 's accuracy={acc * 100:.2f}%")
        # print(f"Emoset label 0={labels_report['0']['precision']*100:.2f}%")
        # print(f"Emosetlabel 1={labels_report['1']['precision']*100:.2f}%")
        # print(f"Emosetlabel 2={labels_report['2']['precision']*100:.2f}%")
        # print(f"Emosetlabel 3={labels_report['3']['precision']*100:.2f}%")
        # print(f"Emosetlabel 4={labels_report['4']['precision']*100:.2f}%")
        # print(f"Emoset label 5={labels_report['5']['precision']*100:.2f}%")
        self.log("Emoset's ACC", acc, logger=True)
        self.log("Emoset label 0", float(labels_report['0']['precision']), logger=True)
        self.log("Emoset label 1", float(labels_report['1']['precision']), logger=True)
        self.log("Emoset label 2", float(labels_report['2']['precision']), logger=True)
        self.log("Emoset label 3", float(labels_report['3']['precision']), logger=True)
        self.log("Emoset label 4", float(labels_report['4']['precision']), logger=True)
        self.log("Emoset label 5", float(labels_report['5']['precision']), logger=True)

        # confusion = metrics.confusion_matrix(labels, pred)
        # print("Confusion Matrix:")
        # print(confusion)
        pred = torch.stack(self.val_ser_all_preds).cpu()
        labels = torch.stack(self.val_ser_all_targets).cpu()
        labels_report = metrics.classification_report(labels, pred, digits=4, output_dict=True)
        acc = metrics.accuracy_score(labels, pred)
        print(f"SER 's accuracy={acc * 100:.2f}%")
        self.log("SER's ACC", acc, logger=True)
        self.log("SER label 0", float(labels_report['0']['precision']), logger=True)
        self.log("SER label 1", float(labels_report['1']['precision']), logger=True)
        self.log("SER label 2", float(labels_report['2']['precision']), logger=True)
        self.log("SER label 3", float(labels_report['3']['precision']), logger=True)
        self.log("SER label 4", float(labels_report['4']['precision']), logger=True)
        self.log("SER label 5", float(labels_report['5']['precision']), logger=True)
        self.val_emo_all_preds = []
        self.val_ser_all_preds = []
        self.val_emo_all_targets = []
        self.val_ser_all_targets = []
        # free memory
        self.val_emo_all_preds.clear()
        self.val_ser_all_preds.clear()
        self.val_emo_all_targets.clear()
        self.val_ser_all_targets.clear()

    def on_test_epoch_end(self):
        pred = torch.stack(self.test_all_preds).cpu()
        labels = torch.stack(self.test_all_targets).cpu()
        report = metrics.classification_report(labels, pred, digits=4)
        print("Classification Report:")
        print(report)
        # confusion = metrics.confusion_matrix(labels, pred)
        # print("Confusion Matrix:")
        # print(confusion)

        self.test_all_preds = []
        self.test_all_targets = []
        # free memory
        self.test_all_preds.clear()
        self.test_all_targets.clear()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path, strict=True):
        # TODO diff be
        self.load_state_dict(torch.load(path)['state_dict'], strict=True)


def build_transform_da(mode):
    if mode == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=512,
            is_training=True,
            color_jitter=0.1,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0,
            re_mode='pixel',
            re_count=1,
        )
        return transform
    else:
        t = [transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        return transforms.Compose(t)




