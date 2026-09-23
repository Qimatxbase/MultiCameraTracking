"""MEVID Dataset loader for image-based ReID."""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from fastreid.data.datasets import DATASET_REGISTRY
    from fastreid.data.datasets.bases import ImageDataset
    HAS_FASTREID = True
except ImportError:
    HAS_FASTREID = False
    ImageDataset = object

def register_mevid():
    if not HAS_FASTREID:
        return lambda x: x
    return DATASET_REGISTRY.register()

@register_mevid()
class MEVID(ImageDataset):
    """MEVID Dataset loader compatible with Fast-ReID.
    
    Parses MEVID annotations and creates training, query, and gallery splits.
    Dataset tuple format: (img_path: str, pid: int, camid: int, outfit_id: int)
    """
    dataset_dir = "mevid"
    dataset_url = ""

    def __init__(self, root="data", **kwargs):
        """Initialize the MEVID dataset."""
        self.root = Path(root)
        self.dataset_dir = self.root / self.dataset_dir
        
        self.annotation_dir = self.dataset_dir / "mevid-v1-annotation-data"
        self.train_dir = self.dataset_dir / "bbox_train"
        self.test_dir = self.dataset_dir / "bbox_test"

        self.train_data: List[Tuple[str, int, int, int]] = []
        self.query_data: List[Tuple[str, int, int, int]] = []
        self.gallery_data: List[Tuple[str, int, int, int]] = []

        self.pid_map: Dict[int, int] = {}
        self.outfit_map: Dict[int, int] = {}

        self._load_dataset()
        
        if HAS_FASTREID:
            super(MEVID, self).__init__(self.train_data, self.query_data, self.gallery_data, **kwargs)
        else:
            self.train = self.train_data
            self.query = self.query_data
            self.gallery = self.gallery_data

    def _load_dataset(self) -> None:
        """Parse annotations and populate dataset splits."""
        if not self.annotation_dir.exists():
            logger.warning(f"Annotation dir not found: {self.annotation_dir}. Dataset is empty.")
            return

        train_names = self._read_lines(self.annotation_dir / "train_name.txt")
        test_names = self._read_lines(self.annotation_dir / "test_name.txt")

        train_info_path = self.annotation_dir / "track_train_info.txt"
        self._process_train_info(train_info_path, train_names)

        test_info_path = self.annotation_dir / "track_test_info.txt"
        query_idx_path = self.annotation_dir / "query_IDX.txt"
        self._process_test_info(test_info_path, query_idx_path, test_names)

    def _read_lines(self, file_path: Path) -> List[str]:
        if not file_path.exists():
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _process_train_info(self, info_path: Path, img_names: List[str]) -> None:
        lines = self._read_lines(info_path)
        pid_counter = 0

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            
            start_idx, end_idx, pid, outfit_id, camid = [int(float(x)) for x in parts]
            
            if pid not in self.pid_map:
                self.pid_map[pid] = pid_counter
                pid_counter += 1
            mapped_pid = self.pid_map[pid]

            for step, i in enumerate(range(start_idx, end_idx + 1)):
                if step % 20 != 0:
                    continue
                
                idx = i - 1 if start_idx > 0 else i
                if idx < 0 or idx >= len(img_names):
                    continue
                
                img_name = img_names[idx]
                img_path = str(self.train_dir / f'{pid:04d}' / img_name)
                if not Path(img_path).exists(): continue
                self.train_data.append((img_path, mapped_pid, camid))

    def _process_test_info(self, info_path: Path, query_path: Path, img_names: List[str]) -> None:
        lines = self._read_lines(info_path)
        query_indices_str = self._read_lines(query_path)
        query_indices = set(int(float(idx)) for idx in query_indices_str)

        for row_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue
            
            start_idx, end_idx, pid, outfit_id, camid = [int(float(x)) for x in parts]

            tracklet_samples = []
            for step, i in enumerate(range(start_idx, end_idx + 1)):
                if step % 20 != 0:
                    continue
                
                idx = i - 1 if start_idx > 0 else i
                if idx < 0 or idx >= len(img_names):
                    continue
                
                img_name = img_names[idx]
                img_path = str(self.test_dir / f'{pid:04d}' / img_name)
                if not Path(img_path).exists(): continue
                tracklet_samples.append((img_path, pid, camid, outfit_id))

            if row_idx in query_indices:
                self.query_data.extend(tracklet_samples)
            else:
                self.gallery_data.extend(tracklet_samples)

    def get_num_pids(self) -> int:
        return len(self.pid_map)

    def get_num_cams(self) -> int:
        cams = set(item[2] for item in self.train_data)
        return len(cams)

# Legacy alias for the rest of the pipeline
MEVIDDataset = MEVID

@register_mevid()
class MEVID_Sample(MEVID):
    """A tiny subset of MEVID for quickly testing the training loop."""
    def _load_dataset(self) -> None:
        super()._load_dataset()
        
        logger.info("Shrinking dataset for MEVID_Sample dry run...")
        
        # 1. Shrink Training (Keep exactly 10 identities, up to 20 images each)
        train_dict = {}
        for item in self.train_data:
            pid = item[1]
            if pid not in train_dict:
                train_dict[pid] = []
            if len(train_dict[pid]) < 20:
                train_dict[pid].append(item)
                
        tiny_train = []
        for pid in list(train_dict.keys())[:10]:
            tiny_train.extend(train_dict[pid])
        self.train_data = tiny_train

        # 2. Shrink Testing (Keep 5 identities)
        query_dict = {}
        for item in self.query_data:
            pid = item[1]
            if pid not in query_dict:
                query_dict[pid] = []
            if len(query_dict[pid]) < 5:
                query_dict[pid].append(item)
                
        tiny_query = []
        sample_test_pids = list(query_dict.keys())[:5]
        for pid in sample_test_pids:
            tiny_query.extend(query_dict[pid])
        self.query_data = tiny_query
        
        # Keep gallery matches for those 5 test identities
        self.gallery_data = [x for x in self.gallery_data if x[1] in sample_test_pids][:200]
        

            
        logger.info(f"Sample loaded: {len(self.train_data)} train, {len(self.query_data)} query, {len(self.gallery_data)} gallery")

