"""Run TG-Net inference and dump prediction maps.

    python test.py --config configs/ham10000.yaml --checkpoint Snapshots/ham10000/TGNet.pth
    python test.py --config configs/ph2.yaml --checkpoint Snapshots/ph2/TGNet.pth

Outputs are 8-bit grayscale PNGs named after the input images, ready to be
scored by ``eval/main.m`` (set ``ResultMapPath`` to this folder).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
import skimage.io as io

from tgnnet.config import load_config, REPO_ROOT
from tgnnet.datasets import build_inference_loader, resolve_spec
from tgnnet.models import build_model
from tgnnet.utils import ensure_dir, normalize_map, stem


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TG-Net inference")
    parser.add_argument("--config", default="configs/ham10000.yaml",
                        help="path to a YAML config")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint")
    parser.add_argument("--testsize", type=int, default=None, help="inference size")
    parser.add_argument("--data_path", default=None,
                        help="override the test split folder")
    parser.add_argument("--save_path", default=None,
                        help="override the prediction output folder")
    parser.add_argument("--branch", default="sup2",
                        choices=["sup1", "sup2", "sup3", "sup4", "mean"],
                        help="which decoder output to export (default: Sup-2, "
                             "the headline prediction in the paper)")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    spec = resolve_spec(cfg.data.name)
    device = torch.device(f"cuda:{args.gpu if args.gpu is not None else cfg.train.get('gpu', 0)}"
                          if torch.cuda.is_available() else "cpu")

    root = cfg.data.get("root", "data")
    root = root if os.path.isabs(root) else os.path.join(REPO_ROOT, root)
    test_dir = args.data_path or cfg.data.get("test_dir") or os.path.join(root, cfg.data.name)
    csv_subdir = cfg.data.get("csv_subdir", "csv")
    testsize = args.testsize or cfg.test.testsize
    checkpoint = args.checkpoint or cfg.test.get("checkpoint")
    save_path = ensure_dir(args.save_path or os.path.join(REPO_ROOT, cfg.test.save_path))

    if not checkpoint or not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}\n"
            f"Train first, or pass --checkpoint /path/to/TGNet.pth"
        )

    print("#" * 24, "TG-Net inference", "#" * 24)
    print(f"config      : {cfg['_config_path']}")
    print(f"dataset     : {spec} | branch = {args.branch}")
    print(f"checkpoint  : {checkpoint}")
    print(f"outputs     : {save_path}")

    loader, dataset = build_inference_loader(
        image_root=os.path.join(test_dir, "images") + os.sep,
        testsize=testsize,
        spec=spec,
        csv_root=os.path.join(test_dir, csv_subdir) if csv_subdir else test_dir,
    )
    print(f"images      : {len(dataset)}")

    model = build_model(cfg, prompt_dim=spec.prompt_dim)
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing key(s), e.g. {missing[:3]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected key(s), e.g. {unexpected[:3]}")

    model = model.to(device).eval()

    with torch.no_grad():
        for images, prompts, names, original_sizes in loader:
            images = images.to(device)
            prompts = prompts.to(device).float()

            sup1, sup2, sup3, sup4 = model(images, prompts)
            branches = {"sup1": sup1, "sup2": sup2, "sup3": sup3, "sup4": sup4}
            if args.branch == "mean":
                # The branches sit at different resolutions; bring them all
                # onto the Sup-1 grid before averaging.
                ref = sup1.shape[-2:]
                aligned = [F.interpolate(s, size=ref, mode="bilinear",
                                         align_corners=False)
                           if s.shape[-2:] != ref else s
                           for s in branches.values()]
                branch_map = sum(aligned) / len(aligned)
            else:
                branch_map = branches[args.branch]

            # Export at the original image size so that eval/main.m can compare
            # against the ground-truth masks without resizing.
            width, height = int(original_sizes[0][0]), int(original_sizes[1][0])
            if branch_map.shape[-2:] != (height, width):
                branch_map = F.interpolate(branch_map, size=(height, width),
                                           mode="bilinear", align_corners=False)

            prob = torch.sigmoid(branch_map)[0, 0].cpu().numpy()
            prob = normalize_map(prob)

            io.imsave(os.path.join(save_path, f"{stem(names[0])}.png"),
                      img_as_ubyte(prob), check_contrast=False)

    print(f"done. {len(dataset)} prediction map(s) written to {save_path}")


if __name__ == "__main__":
    sys.exit(main())
