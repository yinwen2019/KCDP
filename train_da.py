import argparse
import os

import torch
import lightning.pytorch as pl
import yaml
from torch.utils.data import DataLoader

import numpy as np
import datetime

from dataset import SERCaption, Emoset, SERCaptionAligned, EmosetAligned, EmosetAligned_1, SERCaptionAligned_1
from models.kadap_da import KADAPDA

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def main():
    # data parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", default="Emoset", type=str, help="SER, Emoset")
    parser.add_argument("--preprocess_path", default="/home/user/xxx/MultiModal/TGCA_PVT/preprocess", type=str)
    parser.add_argument("--seed", type=int, default=42)

    # debug presets
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--val_debug", action='store_true', default=False)
    parser.add_argument("--wandb_debug", action='store_true', default=False)
    parser.add_argument("--wandb_group", type=str, default="FT_baseline_runs")

    # logger parameter
    parser.add_argument("--exp_name", default="SERTrain", type=str)
    parser.add_argument("--model_name", type=str, default="APKA")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=1)

    # callback/trainer ckpt parameter
    parser.add_argument("--save_top_k", type=int, default=0)
    parser.add_argument("--save_last", default=True)
    parser.add_argument("--check_model_every_n_epochs", type=int, default=-1)
    parser.add_argument('--trainer_ckpt_path', type=str, default=None)

    # model checkpoint parameters
    parser.add_argument("--from_scratch", action='store_true', default=False)
    parser.add_argument("--load_checkpoint_path", type=str, default=None)
    parser.add_argument('--save_checkpoint_path', type=str, default='weights/')

    # trainer parameters
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--limit_train_batches", default=0.5, help="How much of training dataset to check.")
    parser.add_argument("--limit_val_batches", default=None, help="How much of validation dataset to check.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)

    # APKA model parameters
    parser.add_argument('--freeze_backbone', type=int, default=0)
    parser.add_argument("--freeze_text_adapter", type=int, default=1)
    parser.add_argument('--text_conditioning', type=str, default='blip+triple')
    parser.add_argument('--use_scaled_encode', action='store_true', default=False)
    parser.add_argument('--append_self_attention', action='store_true', default=False)
    parser.add_argument('--cond_stage_trainable', action='store_true', default=False)
    parser.add_argument('--no_attn', action='store_true', default=False)
    parser.add_argument('--use_only_attn', action='store_true', default=False)
    args = parser.parse_args()

    # ---debug preset
    wandb_group = args.wandb_group  # if debug change
    # logger
    wandb_name = args.exp_name + '_' + args.model_name  # if debug change
    log_freq = args.log_freq  # if debug change
    log_every_n_steps = args.log_every_n_steps
    # callback
    check_model_every_n_epochs = args.check_model_every_n_epochs
    save_top_k = args.save_top_k  # if debug change
    save_last = args.save_last  # if debug change
    trainer_ckpt_path = args.trainer_ckpt_path
    # trainer
    max_epochs = args.max_epochs  # if debug change
    num_workers = args.num_workers  # if debug change
    limit_train_batches = args.limit_train_batches  # if debug change
    limit_val_batches = args.limit_val_batches  # if debug change
    batch_size = args.batch_size  # if debug change
    val_batch_size = args.val_batch_size

    online = False
    if not online:
        os.environ["WANDB_MODE"] = "offline"
    if args.debug:
        max_epochs = 4
        limit_val_batches = 0.01
        limit_train_batches = 0.01
        os.environ["WANDB_MODE"] = "dryrun"
        log_freq = 1
        save_last = False
        save_top_k = 0
    if args.wandb_debug:
        wandb_group = "wandb_debugging"
        wandb_name = f"dummy_{datetime.datetime.now().__str__()}"
        log_freq = 1
        save_last = False
        save_top_k = 0
    if args.val_debug:
        limit_val_batches = 0.01
        limit_train_batches = 0.01
        os.environ["WANDB_MODE"] = "dryrun"

    # checkpoint
    pretrained = not args.from_scratch
    load_checkpoint_path = args.load_checkpoint_path
    save_checkpoint_path = args.save_checkpoint_path

    pl.seed_everything(args.seed)

    # init dataset
    train_s_dataset = None
    train_t_dataset = None
    if args.train_dataset == 'Emoset':
        train_s_dataset = EmosetAligned_1(preprocess_path=args.preprocess_path, mode='train')
        train_t_dataset = SERCaptionAligned_1(preprocess_path=args.preprocess_path, mode='train')
    elif args.train_dataset == 'SER':
        train_s_dataset = SERCaptionAligned_1(preprocess_path=args.preprocess_path, mode='train')
        train_t_dataset = EmosetAligned_1(preprocess_path=args.preprocess_path, mode='train', data_len=10000)
        limit_train_batches = None

    val_emo_dataset = EmosetAligned_1(preprocess_path=args.preprocess_path, mode='val')
    val_ser_dataset = SERCaptionAligned_1(preprocess_path=args.preprocess_path, mode='val')

    # load data
    train_loaders = [DataLoader(train_s_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                              drop_last=True)] + [
                     DataLoader(train_t_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                                drop_last=True)]

    val_loaders = [DataLoader(val_emo_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                              drop_last=True)] + [
                     DataLoader(val_ser_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                                drop_last=True)]
    cfg = yaml.load(open("./sd_tune.yaml", "r"), Loader=yaml.FullLoader)
    cfg["annotator"]["type"] = "ground_truth"
    cfg["stable_diffusion"]["use_diffusion"] = True
    cfg["max_epochs"] = max_epochs
    cfg["freeze_text_adapter"] = args.freeze_text_adapter

    if args.no_attn and args.use_only_attn:
        raise ValueError('Cannot use both no_attn and use_only_attn')

    cfg['text_conditioning'] = args.text_conditioning
    cfg['use_scaled_encode'] = args.use_scaled_encode
    cfg['append_self_attention'] = args.append_self_attention
    cfg['cond_stage_trainable'] = args.cond_stage_trainable
    cfg['use_attn'] = not args.no_attn
    cfg['use_only_attn'] = args.use_only_attn
    cfg['dataset_len'] = len(train_s_dataset)
    cfg['preprocess_path'] = args.preprocess_path
    # init model
    model = KADAPDA(cfg=cfg, train_dataset=args.train_dataset, freeze_backbone=args.freeze_backbone)

    # init model checkpoint
    if load_checkpoint_path is not None and pretrained:
        try:
            state_dict = torch.load(load_checkpoint_path)["state_dict"]
            # making older state dicts compatible with current model
            if list(state_dict.keys())[0] != list(model.state_dict().keys())[0]:
                print('Loading pretrained model with different key names')
                # replace each key in state_dict with the corresponding key in model.state_dict()
                state_dict = {list(model.state_dict().keys())[i]: list(state_dict.values())[i] for i in
                              range(len(state_dict))}
            model.load_state_dict(state_dict, strict=True)
        except KeyError:
            model.load_state_dict(torch.load(load_checkpoint_path))

    # init trainer ckpt
    checkpoint_callbacks = []
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=f'val_loss',
        dirpath=f'checkpoints/{wandb_name}/',
        save_top_k=-1 if check_model_every_n_epochs > 0 else save_top_k,
        mode='min',  # Mode for comparing the monitored metric
        save_last=save_last,
        every_n_epochs=check_model_every_n_epochs if check_model_every_n_epochs > 0 else None,
    )
    checkpoint_callbacks.append(checkpoint_callback)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_callback] + checkpoint_callbacks

    # init logger
    logger = pl.loggers.WandbLogger(
        name=wandb_name,
        group=wandb_group,
        project="APKA",
        log_model="all",
    )
    logger.watch(model, log="all", log_freq=log_freq)

    # init trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[2],
        #devices=[0, 1, 2, 3],
        logger=logger,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        limit_train_batches=limit_train_batches,  # None unless --wandb_debug or --val_debug flag is set
        limit_val_batches=limit_val_batches,  # None unless --wandb_debug or --val_debug flag is set
        sync_batchnorm=True if args.num_gpus > 1 else False,
        # accum_grad_batches=1
    )
    if trainer.global_rank == 0:
        logger.experiment.config.update(args)

    if not args.debug or args.val_debug:
        trainer.validate(model, dataloaders=val_loaders)

    trainer.fit(
        model,
        train_dataloaders=train_loaders,
        val_dataloaders=val_loaders,
        ckpt_path=trainer_ckpt_path,
    )
    # save the model
    if save_checkpoint_path != '':
        save_model_name = f'{wandb_name + datetime.datetime.now().__str__()}.ckpt'
        # results paths
        if not os.path.exists(save_checkpoint_path):
            os.makedirs(save_checkpoint_path)
        torch.save(model.state_dict(), save_checkpoint_path + save_model_name)


if __name__ == "__main__":
    main()
