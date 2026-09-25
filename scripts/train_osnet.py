import argparse
import collections
import collections.abc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fast-reid" / "upstream"))
os.environ["FASTREID_DATASETS"] = str(ROOT / "data")
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights" / "torch"))


def load_splits():
    """Read inclusive, zero-based MEVID tracklets; retain every twentieth frame."""
    from tqdm import tqdm

    root = ROOT / "data" / "mevid"
    annotations = root / "mevid-v1-annotation-data"
    query_ids = {int(float(x)) for x in (annotations / "query_IDX.txt").read_text().split()}
    splits = {"train": [], "query": [], "gallery": []}
    for subset in ("train", "test"):
        names = (annotations / f"{subset}_name.txt").read_text().splitlines()
        tracks = (annotations / f"track_{subset}_info.txt").read_text().splitlines()
        for row, line in enumerate(tqdm(tracks, desc=f"Loading {subset} tracklets", unit="track", dynamic_ncols=True)):
            start, end, pid, outfit, camera = map(lambda x: int(float(x)), line.split())
            if end == start - 1:
                continue  # MEVID includes empty tracklets; keep original row numbers.
            if not 0 <= start <= end < len(names):
                raise ValueError(f"Invalid {subset} tracklet {row}: {start}, {end}")
            split = "train" if subset == "train" else ("query" if row in query_ids else "gallery")
            for index in range(start, end + 1, 20):
                path = root / f"bbox_{subset}" / f"{pid:04d}" / names[index].strip()
                if not path.is_file():
                    raise FileNotFoundError(path)
                splits[split].append((str(path), pid, camera))
    for name, items in splits.items():
        if not items:
            raise ValueError(f"Empty MEVID {name} split")
        print(f"{name}: {len(items)} images, {len({x[1] for x in items})} identities", flush=True)
    return splits


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: python scripts/train_osnet.py --epochs 60 --batch-size 64",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--resume", action="store_true", help="resume the latest saved epoch")
    parser.add_argument("--check-data", action="store_true", help="validate data without training")
    parser.add_argument("--epochs", type=positive_int, help="total epochs (otherwise use config)")
    parser.add_argument("--batch-size", type=positive_int, help="training batch size (multiple of 4)")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="training device (otherwise use config)")
    parser.add_argument("--output-dir", help="checkpoint and metrics directory")
    parser.add_argument("--log-every", type=positive_int, default=10, help="write metrics every N batches")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="advanced FastReID KEY VALUE overrides")
    args = parser.parse_args()
    print("OSNet fine-tuning | press Ctrl+C to stop", flush=True)
    print(f"Dataset: {ROOT / 'data' / 'mevid'}", flush=True)
    splits = load_splits()
    if args.check_data:
        return

    print("Loading PyTorch and FastReID...", flush=True)

    # Compatibility with FastReID's older Python imports, local to this runner.
    collections.Mapping = collections.abc.Mapping
    collections.Iterable = collections.abc.Iterable
    import torch
    from fastreid.config import get_cfg
    from fastreid.data.datasets import DATASET_REGISTRY
    from fastreid.data.datasets.bases import ImageDataset
    from fastreid.engine import DefaultTrainer, default_setup
    from fastreid.engine import hooks
    from fastreid.engine.train_loop import HookBase
    from fastreid.evaluation import ReidEvaluator
    from tqdm import tqdm

    class TerminalProgress(HookBase):
        bar = None

        def before_epoch(self):
            self.bar = tqdm(
                total=self.trainer.iters_per_epoch,
                desc=f"Epoch {self.trainer.epoch + 1}/{self.trainer.max_epoch}",
                unit="batch", dynamic_ncols=True,
            )

        def after_step(self):
            latest = self.trainer.storage.latest()
            loss = latest.get("total_loss")
            if loss is not None:
                self.bar.set_postfix(loss=f"{loss[0]:.4f}", refresh=False)
            self.bar.update(1)

        def after_epoch(self):
            if self.bar is not None:
                self.bar.close()

        def after_train(self):
            if self.bar is not None:
                self.bar.close()

    class OSNetEvaluator(ReidEvaluator):
        def _compile_dependencies(self):
            # Use the upstream NumPy rank evaluator without a Windows C compiler.
            pass

    class OSNetTrainer(DefaultTrainer):
        def build_hooks(self):
            training_hooks = super().build_hooks()
            for hook in training_hooks:
                if isinstance(hook, hooks.PeriodicWriter):
                    hook._period = args.log_every
            return [TerminalProgress(), *training_hooks]

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_dir=None):
            loader, num_query = cls.build_test_loader(cfg, dataset_name)
            return loader, OSNetEvaluator(cfg, num_query, output_dir)

    @DATASET_REGISTRY.register()
    class MEVID_OSNet(ImageDataset):
        def __init__(self, root=None, **kwargs):
            super().__init__(splits["train"], splits["query"], splits["gallery"], **kwargs)

    cfg = get_cfg()
    cfg.merge_from_file(str(ROOT / "configs" / "mevid_osnet.yml"))
    cfg.merge_from_list(args.opts)
    if args.epochs is not None:
        cfg.SOLVER.MAX_EPOCH = args.epochs
    if args.batch_size is not None:
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    if args.device is not None:
        cfg.MODEL.DEVICE = args.device
    if args.output_dir is not None:
        cfg.OUTPUT_DIR = args.output_dir
    if cfg.SOLVER.IMS_PER_BATCH % cfg.DATALOADER.NUM_INSTANCE:
        parser.error("batch size must be divisible by DATALOADER.NUM_INSTANCE")
    if cfg.SOLVER.IMS_PER_BATCH > len(splits["train"]):
        parser.error("batch size must not exceed the training split size")
    if cfg.MODEL.DEVICE == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation; use --device cpu")
    if cfg.MODEL.BACKBONE.NAME != "build_osnet_backbone":
        raise ValueError("This runner trains OSNet only")
    cfg.OUTPUT_DIR = str(ROOT / cfg.OUTPUT_DIR)
    if cfg.MODEL.DEVICE == "cpu":
        # FastReID's prefetch loader allocates CUDA streams even on CPU.
        import fastreid.data.build as data_build

        def cpu_loader(local_rank, **kwargs):
            kwargs["pin_memory"] = False
            kwargs["num_workers"] = 0
            return torch.utils.data.DataLoader(**kwargs)

        data_build.DataLoaderX = cpu_loader
    cfg.freeze()
    args.config_file = str(ROOT / "configs" / "mevid_osnet.yml")
    args.eval_only = False
    default_setup(cfg, args)
    trainer = OSNetTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    print(
        f"Training on {cfg.MODEL.DEVICE} | {cfg.SOLVER.MAX_EPOCH} epochs | "
        f"batch size {cfg.SOLVER.IMS_PER_BATCH}\nOutputs: {cfg.OUTPUT_DIR}",
        flush=True,
    )
    metrics = trainer.train()
    result_path = Path(cfg.OUTPUT_DIR) / "evaluation.json"
    result_path.write_text(json.dumps(metrics, indent=2, default=lambda x: x.item()), encoding="utf-8")
    print(f"OSNet evaluation saved to {result_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Use --resume to continue from the latest saved epoch.", flush=True)
        sys.exit(130)
