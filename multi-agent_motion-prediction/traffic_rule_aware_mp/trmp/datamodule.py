from typing import Iterator, Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class DataConfig:
    name: str
    root: str
    # for nuScenes
    sample_token: Optional[str] = None
    # for AV2
    av2_log_dir: Optional[str] = None
    # for Waymo
    waymo_tfrecord: Optional[str] = None

class DataModule:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        if self.cfg.name.lower() == "nuscenes":
            from .datasets.nuscenes import load_sample
            if not self.cfg.sample_token:
                raise ValueError("nuscenes requires sample_token")
            yield load_sample(self.cfg.root, self.cfg.sample_token)
        elif self.cfg.name.lower() in ["av2", "argoverse2", "av2sensor"]:
            from .datasets.av2sensor import load_log
            if not self.cfg.av2_log_dir:
                raise ValueError("av2sensor requires av2_log_dir")
            yield load_log(self.cfg.av2_log_dir)
        elif self.cfg.name.lower() in ["waymo", "waymo_perception"]:
            from .datasets.waymo_perception import load_first_frame_with_maps
            if not self.cfg.waymo_tfrecord:
                raise ValueError("waymo_perception requires waymo_tfrecord")
            yield load_first_frame_with_maps(self.cfg.waymo_tfrecord)
        else:
            raise ValueError(f"Unknown dataset name: {self.cfg.name}")
