from typing import Dict, Any
from pathlib import Path

def _optional_import():
    try:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2 as open_dataset
        return tf, open_dataset
    except Exception as e:
        raise ImportError("Waymo TFRecord loader requires TensorFlow 2.11 and waymo-open-dataset wheel.") from e

def iter_segment(tfrecord_path: str):
    tf, open_dataset = _optional_import()
    for data in tf.data.TFRecordDataset(tfrecord_path, compression_type=''):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        yield frame

def load_first_frame_with_maps(tfrecord_path: str) -> Dict[str, Any]:
    """Return a unified sample dict from the first frame of a Waymo Perception TFRecord.
    NOTE: Waymo Perception with maps is required (v1.4.2/1.4.3). This skeleton returns image paths as None
    (images are embedded bytes) and leaves map decoding to user (map protos vary).
    """
    tf, open_dataset = _optional_import()
    for frame in iter_segment(tfrecord_path):
        # 1) Images: we output decoded bytes per camera as a placeholder
        images = {}
        for img in frame.images:
            cam_name = open_dataset.CameraName.Name.Name(img.name)
            images[cam_name] = None  # user can decode JPEG bytes from img.image if needed

        # 2) LiDAR: Waymo provides range images; converting to XYZ requires additional code.
        lidar = {"range_images": True}

        # 3) Agents history: requires iterating the whole segment & collecting labels by track_id.
        agents_history = {}

        # 4) Map: available in map-enabled Perception; user should parse frame.map_features
        map_layers = {"lane_center": [], "lane_boundary": [], "road_boundary": [], "crosswalk": [], "stop_line_or_sign": []}

        return {
            "images": images,
            "lidar": lidar,
            "calib": {},
            "ego_history": None,
            "agents_history": agents_history,
            "map": map_layers,
            "timestamps": {"t0": int(frame.timestamp_micros), "history_hz": 10.0, "lidar_hz": 10.0, "cam_hz": 10.0},
            "city_or_map_id": "",
        }
    raise RuntimeError("Empty TFRecord or failed to parse any frame.")
