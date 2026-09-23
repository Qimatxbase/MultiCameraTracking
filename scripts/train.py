"""
Training entrypoint script for ReID models.
"""

import argparse
import logging
import sys
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable
import torch
torch.backends.cudnn.enabled = False

try:
    import fastreid
except ImportError:
    print("Error: fast-reid is not installed. Please install it first.", file=sys.stderr)
    sys.exit(1)

from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, launch, DefaultTrainer

# Import our custom dataset to trigger Fast-ReID registration
from src.data.mevid_dataset import MEVID

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)
        
        # Load weights and evaluate
        # Using DefaultTrainer evaluation logic if implemented
        return 
        
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--config", dest="config_file", required=True, help="path to config file")
    parser.add_argument("--model", type=str, help="Model type (e.g. resnet50)")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
