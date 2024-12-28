import os
import modal.gpu
import yaml
import sys
import json
from argparse import ArgumentParser
import numpy as np


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

import modal
app = modal.App(
    image=modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("libgl1", "libglib2.0-0", "libxrender1")
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
    .copy_local_dir("models", "/root/models")
    .workdir("/root/models/ops")
    .run_commands(["python setup.py install"], gpu=modal.gpu.A10G(count=1))
    .workdir("/root")
    .copy_local_dir("configs", "/root/configs")
    .add_local_python_source("dataset_road_network", "evaluator", "train", "trainer", "utils", "losses", "metric_map", "metric_smd", "metric_topo", "box_ops_2D", "inference", "models")
)
backend_secrets = modal.Secret.from_name("grail-backend-secrets")
s3_creds = modal.Secret.from_name("aws-s3-secret")
s3_bucket_name = "grail-pilot-storage"
s3_vol = modal.CloudBucketMount(s3_bucket_name, secret=s3_creds)
s3_mount_path = f"/{s3_bucket_name}"
vol = modal.Volume.from_name("trained_weights", create_if_missing=True)
vol_mountpath = "/trained_weights"
@app.function(gpu=modal.gpu.A10G(count=1), volumes={vol_mountpath: vol, s3_mount_path: s3_vol}, timeout=86400)
def main(config=None, resume=None, device='cuda'):
    import logging
    import ignite
    import torch
    import torch.nn as nn
    from monai.data import DataLoader
    from monai.engines import SupervisedTrainer
    from monai.handlers import MeanDice, StatsHandler
    from monai.inferers import SimpleInferer
    from dataset_road_network import build_road_network_data
    from evaluator import build_evaluator
    from trainer import build_trainer
    from models import build_model
    from monai.losses import DiceCELoss
    from utils import image_graph_collate_road_network
    from tensorboardX import SummaryWriter
    from models.matcher import build_matcher
    from losses import SetCriterion


    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Load the config files
    with open(config) as f:
        print('\n*** Config file')
        print(config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda") if device=='cuda' else torch.device("cpu")

    net = build_model(config).to(device)

    matcher = build_matcher(config)
    loss = SetCriterion(config, matcher, net)

    train_ds, val_ds = build_road_network_data(
        config, mode='split'
    )

    train_loader = DataLoader(train_ds,
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=True,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_road_network,
                            pin_memory=True)

    val_loader = DataLoader(val_ds,
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_road_network,
                            pin_memory=True)

    param_dicts = [
        {
            "params":
                [p for n, p in net.named_parameters()
                 if not match_name_keywords(n, ["encoder.0"]) and not match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
            "lr": float(config.TRAIN.LR)
        },
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ["encoder.0"]) and p.requires_grad],
            "lr": float(config.TRAIN.LR_BACKBONE)
        },
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
            "lr": float(config.TRAIN.LR) * 0.1
        }
    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.LR_DROP)

    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        last_epoch = scheduler.last_epoch

    writer = SummaryWriter(
        log_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs_2", '%s_%d' % (config.log.exp_name, config.DATA.SEED)),
    )

    evaluator = build_evaluator(
        val_loader,
        net, optimizer,
        scheduler,
        writer,
        config,
        device
    )
    trainer = build_trainer(
        train_loader,
        net,
        loss,
        optimizer,
        scheduler,
        writer,
        evaluator,
        config,
        device,
        # fp16=fp16,
    )

    if resume:
        evaluator.state.epoch = last_epoch
        trainer.state.epoch = last_epoch
        trainer.state.iteration = trainer.state.epoch_length * last_epoch

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    trainer.run()


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out