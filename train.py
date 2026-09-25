"""Train TG-Net.

    python train.py --config configs/ham10000.yaml
    python train.py --config configs/ph2.yaml --epoch 30 --gpu 1

Everything is driven by the YAML config; the flags below are convenience
overrides for the keys you change most often.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tgnnet.config import load_config, REPO_ROOT
from tgnnet.datasets import build_dataloader, resolve_spec
from tgnnet.losses import multi_branch_structure_loss
from tgnnet.models import build_model
from tgnnet.utils import (AvgMeter, adjust_lr, clip_gradient, count_parameters,
                          ensure_dir, set_seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train TG-Net")
    parser.add_argument("--config", default="configs/ham10000.yaml",
                        help="path to a YAML config")
    parser.add_argument("--epoch", type=int, default=None, help="override epochs")
    parser.add_argument("--lr", type=float, default=None, help="override learning rate")
    parser.add_argument("--batchsize", type=int, default=None, help="override batch size")
    parser.add_argument("--trainsize", type=int, default=None, help="override training size")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index")
    parser.add_argument("--save_dir", default=None, help="override snapshot directory")
    parser.add_argument("--pretrained_backbone", default=None,
                        help="path to a res2net50_v1b_26w_4s checkpoint")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    return parser.parse_args(argv)


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, cfg.train.lr)


def run_epoch(loader, model, optimizer, cfg, epoch, device, global_step):
    """One multi-scale training epoch. Returns the mean Sup-1 loss."""
    model.train()
    size_rates = cfg.train.get("size_rates") or [1.0]
    meters = [AvgMeter() for _ in range(4)]
    loss_list = []

    for i, batch in enumerate(loader, start=1):
        if len(batch) == 3:
            images, gts, prompts = batch
        else:  # prompts disabled
            images, gts = batch
            prompts = torch.zeros(images.size(0), 0)

        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        prompts = prompts.to(device, non_blocking=True).float()

        for rate in size_rates:
            optimizer.zero_grad()

            if rate != 1.0:
                trainsize = int(round(cfg.train.trainsize * rate / 32) * 32)
                images_r = F.interpolate(images, size=(trainsize, trainsize),
                                         mode="bilinear", align_corners=True)
                gts_r = F.interpolate(gts, size=(trainsize, trainsize),
                                      mode="bilinear", align_corners=True)
            else:
                images_r, gts_r = images, gts

            predictions = model(images_r, prompts, output_size=gts_r.shape[-2:])
            loss, per_branch = multi_branch_structure_loss(predictions, gts_r)

            loss.backward()
            clip_gradient(optimizer, cfg.train.clip)
            optimizer.step()

            if rate == 1.0:
                for meter, value in zip(meters, per_branch):
                    meter.update(value.item(), images.size(0))
                loss_list.append(per_branch[0].item())

        global_step += 1
        log_every = cfg.train.get("log_every", 20)
        if i % log_every == 0 or i == len(loader):
            print("{:%Y-%m-%d %H:%M:%S} Epoch [{:03d}/{:03d}] Step [{:04d}/{:04d}] "
                  "[Sup-1: {:.4f}, Sup-2: {:.4f}, Sup-3: {:.4f}, Sup-4: {:.4f}]".format(
                      datetime.now(), epoch, cfg.train.epochs, i, len(loader),
                      meters[0].show(), meters[1].show(),
                      meters[2].show(), meters[3].show()))

    return float(np.mean(loss_list)), meters, global_step


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    # ---- command-line overrides -------------------------------------------- #
    for key in ("epoch", "lr", "batchsize", "trainsize", "gpu", "seed",
                "save_dir", "pretrained_backbone"):
        value = getattr(args, key)
        if value is not None:
            cfg["train" if key not in ("pretrained_backbone",) else "model"][key] = value

    seed = cfg.train.get("seed", 42)
    set_seed(seed)

    device = torch.device(f"cuda:{cfg.train.get('gpu', 0)}"
                          if torch.cuda.is_available() else "cpu")
    spec = resolve_spec(cfg.data.name)

    save_path = ensure_dir(os.path.join(REPO_ROOT, cfg.train.get("save_dir", "Snapshots"),
                                        cfg.get("experiment", cfg.data.name)))
    log_path = os.path.join(save_path, "train_log.txt")

    print("#" * 24, "TG-Net training", "#" * 24)
    print(f"config      : {cfg['_config_path']}")
    print(f"dataset     : {spec} | fields = {spec.fields}")
    print(f"device      : {device} | seed = {seed}")
    print(f"epochs      : {cfg.train.epochs} | batch = {cfg.train.batchsize} "
          f"| trainsize = {cfg.train.trainsize}")
    print(f"snapshots   : {save_path}")

    # ---- data -------------------------------------------------------------- #
    root = cfg.data.get("root", "data")
    root = root if os.path.isabs(root) else os.path.join(REPO_ROOT, root)
    train_dir = cfg.data.get("train_dir") or os.path.join(root, cfg.data.name)
    csv_subdir = cfg.data.get("csv_subdir", "csv")

    train_loader, train_set = build_dataloader(
        image_root=os.path.join(train_dir, "images") + os.sep,
        mask_root=os.path.join(train_dir, "masks") + os.sep,
        trainsize=cfg.train.trainsize,
        spec=spec,
        csv_root=os.path.join(train_dir, csv_subdir) if csv_subdir else train_dir,
        batchsize=cfg.train.batchsize,
        shuffle=True,
        num_workers=cfg.train.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        split="train",
    )
    print(f"train pairs : {len(train_set)}")

    # ---- model ------------------------------------------------------------- #
    model = build_model(cfg, prompt_dim=spec.prompt_dim,
                        pretrained_backbone=cfg.model.get("pretrained_backbone"))
    model = model.to(device)

    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
        start_epoch = int(state.get("epoch", 0)) + 1 if isinstance(state, dict) else 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(f"parameters  : {model.n_params / 1e6:.2f} M (trainable)")

    optimizer = build_optimizer(model, cfg)

    # ---- training loop ------------------------------------------------------ #
    global_step, history = 0, []
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} "
                  f"config={cfg['_config_path']} seed={seed} ===\n")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        lr = adjust_lr(optimizer, cfg.train.lr, epoch,
                       cfg.train.get("decay_rate", 0.05),
                       cfg.train.get("decay_epoch", 25))
        t0 = time.time()
        mean_loss, meters, global_step = run_epoch(
            train_loader, model, optimizer, cfg, epoch, device, global_step)
        history.append(mean_loss)

        message = (f"epoch {epoch:03d} done | lr {lr:.2e} | "
                   f"mean Sup-1 loss {mean_loss:.4f} | {time.time() - t0:.1f}s")
        print(message)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(message + "\n")

        if epoch % cfg.train.get("save_every", 10) == 0 or epoch == cfg.train.epochs:
            ckpt = os.path.join(save_path, "TGNet.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "config": cfg["_config_name"], "loss": mean_loss}, ckpt)
            np.savetxt(os.path.join(save_path, "train_loss.csv"),
                       np.asarray(history), delimiter=",")
            print(f"[saved] {ckpt}")

    print("training finished.")


if __name__ == "__main__":
    sys.exit(main())
