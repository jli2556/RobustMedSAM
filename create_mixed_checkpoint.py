"""
Create the RobustMedSAM initialization checkpoint via module-wise fusion.

As described in the paper, RobustMedSAM is initialized by combining two
pretrained ViT-B backbones under the shared SAM architecture:

    image encoder  + prompt encoder  <-  MedSAM      (medical-domain priors)
    mask decoder                     <-  RobustSAM   (corruption robustness)

We start from the RobustSAM state dict (which already contains every
mask-decoder key, including RobustSAM's anti-degradation modules) and overwrite
the image-encoder and prompt-encoder weights with MedSAM's. The resulting
checkpoint is passed to `train_ddp.py --load_model`, after which only the mask
decoder is fine-tuned with the clean-degraded consistency objective.

Usage:
    python create_mixed_checkpoint.py \
        --medsam     medsam_vit_b.pth \
        --robustsam  robustsam_checkpoint_b.pth \
        --output     robustmedsam_init_b.pth
"""
import argparse
from collections import OrderedDict

import torch


def strip_module_prefix(state_dict):
    """Remove a leading 'module.' (DDP) prefix from every key."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k[7:] if k.startswith("module.") else k] = v
    return new_sd


def create_mixed_checkpoint(medsam_path, robustsam_path, output_path):
    print(f"Loading MedSAM (image + prompt encoder) from {medsam_path} ...")
    medsam_sd = strip_module_prefix(
        torch.load(medsam_path, map_location="cpu", weights_only=True)
    )

    print(f"Loading RobustSAM (mask decoder) from {robustsam_path} ...")
    robustsam_sd = strip_module_prefix(
        torch.load(robustsam_path, map_location="cpu", weights_only=True)
    )

    # Start from RobustSAM: it carries all decoder keys (incl. anti-degradation
    # modules). Then overwrite the encoder/prompt-encoder with MedSAM weights.
    merged = OrderedDict(robustsam_sd)

    n_enc, n_prompt = 0, 0
    for k, v in medsam_sd.items():
        if k.startswith("image_encoder."):
            merged[k] = v
            n_enc += 1
        elif k.startswith("prompt_encoder."):
            merged[k] = v
            n_prompt += 1

    print(
        f"Merged: {n_enc} image-encoder + {n_prompt} prompt-encoder tensors from "
        f"MedSAM; mask decoder kept from RobustSAM."
    )

    torch.save(merged, output_path)
    print(f"Saved RobustMedSAM initialization checkpoint to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the RobustMedSAM init checkpoint (MedSAM encoder + RobustSAM decoder)."
    )
    parser.add_argument("--medsam", required=True, help="Path to MedSAM ViT-B checkpoint")
    parser.add_argument("--robustsam", required=True, help="Path to RobustSAM ViT-B checkpoint")
    parser.add_argument("--output", default="robustmedsam_init_b.pth", help="Output checkpoint path")
    args = parser.parse_args()

    create_mixed_checkpoint(args.medsam, args.robustsam, args.output)
