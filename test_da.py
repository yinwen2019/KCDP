import argparse
import csv
import os

import torch
import lightning.pytorch as pl
import yaml
from torch.utils.data import DataLoader, RandomSampler

import numpy as np
import datetime

from dataset import SERCaption, Emoset, EmosetAligned_1, SERCaptionAligned_1, Emo8Aligned_1, EmotionROIAligned_1
from models.kadap import APKA
from models.kadap_da import KADAPDA


def test(model, dataset, ckpt):
    #model = APKA.load_from_checkpoint(ckpt)
    v_file = '/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_CLIEA_ser_v_1k.csv'
    with open(v_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'feature_{i}' for i in range(6)])

    l_file = '/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_CLIEA_ser_l_1k.csv'
    with open(l_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow('y')
    model_weight = torch.load(ckpt)
    model.load_state_dict(model_weight, strict=False)
    subset_size = 1000
    sampler = RandomSampler(dataset, num_samples=subset_size, replacement=False)
    test_dataloader = DataLoader(dataset, sampler=sampler, batch_size=16, num_workers=16)
    trainer = pl.Trainer(accelerator="gpu", devices=[2])
    trainer.test(apka, dataloaders=test_dataloader)


if __name__ == "__main__":
    # data parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", default="SER", type=str, help="SER, Emoset, Emo8, FI, EmotionROI")
    parser.add_argument("--model_ckpt", default="/home/user/xxx/MultiModal/TGCA_PVT/weights/SERTrain_APKA2024-10-08 20:34:02.025197.ckpt", type=str,
                        help="path to model checkpoint")
    parser.add_argument("--preprocess_path", default="/home/user/xxx/MultiModal/TGCA_PVT/preprocess", type=str)

    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument('--freeze_backbone', type=int, default=0)
    parser.add_argument("--freeze_text_adapter", type=int, default=1)
    parser.add_argument('--text_conditioning', type=str, default='blip+triple')
    parser.add_argument('--use_scaled_encode', action='store_true', default=False)
    parser.add_argument('--append_self_attention', action='store_true', default=False)
    parser.add_argument('--cond_stage_trainable', action='store_true', default=False)
    parser.add_argument('--no_attn', action='store_true', default=False)
    parser.add_argument('--use_only_attn', action='store_true', default=False)
    args = parser.parse_args()
    test_dataset = args.test_dataset
    if test_dataset == 'Emoset':
        test_dataset = EmosetAligned_1(preprocess_path=args.preprocess_path, mode='test')
    elif test_dataset == 'SER':
        test_dataset = SERCaptionAligned_1(preprocess_path=args.preprocess_path, mode='test')
    elif test_dataset == 'Emo8':
        test_dataset = Emo8Aligned_1(preprocess_path=args.preprocess_path, mode='test')
    elif test_dataset == 'EmotionROI':
        test_dataset = EmotionROIAligned_1(preprocess_path=args.preprocess_path, mode='test')
    cfg = yaml.load(open("./sd_tune.yaml", "r"), Loader=yaml.FullLoader)
    cfg["annotator"]["type"] = "ground_truth"
    cfg["stable_diffusion"]["use_diffusion"] = True
    cfg["max_epochs"] = args.max_epochs
    cfg["freeze_text_adapter"] = args.freeze_text_adapter
    cfg['text_conditioning'] = args.text_conditioning
    cfg['use_scaled_encode'] = args.use_scaled_encode
    cfg['append_self_attention'] = args.append_self_attention
    cfg['cond_stage_trainable'] = args.cond_stage_trainable
    cfg['use_attn'] = not args.no_attn
    cfg['use_only_attn'] = args.use_only_attn
    cfg['dataset_len'] = len(test_dataset)
    cfg['preprocess_path'] = args.preprocess_path
    if args.no_attn and args.use_only_attn:
        raise ValueError('Cannot use both no_attn and use_only_attn')

    model_ckpt = args.model_ckpt
    apka = KADAPDA(cfg=cfg, freeze_backbone=args.freeze_backbone)
    test(apka, test_dataset, model_ckpt)
