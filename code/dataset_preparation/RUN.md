
# List sample_token with enough history (key frames)
# cd /home/patrick/MSCV_Capstone_Project/code/traffic_rule_aware_mp_v2
# export PYTHONPATH=$PWD
# python scripts/list_nuscenes_sample_tokens.py --dataroot /data/Datasets/NuScenes-QA/data/nuScenes --n 5
python scripts/find_tokens_with_past.py \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --version v1.0-trainval \
  --seconds 4 \
  --n 10

# Generate BEV preview (including multi-agent history)
export PYTHONPATH=$PWD

python scripts/demo_nuscenes_v4.py \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --sample_token 1e19d0a5189b46f4b62aa47508f2983e \
  --with_lidar \
  --past_sec 3 \
  --future_sec 3 \
  --map_radius_m 80 \
  --out_png ./tmp/preview_v17.png


# Packer
python scripts/pack_nuscenes.py \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --version v1.0-trainval \
  --out_dir /data/patrick/packed_nuscenes_v1 \
  --past_sec 3.0 \
  --future_sec 3.0 \
  --stride_sec 0.5 \
  --map_radius_m 80.0 \
  --keep_prefix vehicle. human.pedestrian \
  --min_future_sec 3.0 \
  --compress

python scripts/pack_nuscenes_parallel.py \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --version v1.0-trainval \
  --out_dir /data/patrick/packed_nuscenes_v2_parallel \
  --past_sec 3.0 \
  --future_sec 3.0 \
  --stride_sec 0.5 \
  --map_radius_m 80.0 \
  --keep_prefix vehicle. human.pedestrian \
  --min_future_sec 3.0 \
  --compress \
  --workers 8

# Sanity Check .npz

## Visualization
python scripts/check_npz_preview.py \
  --manifest /data/patrick/packed_nuscenes_v1/manifest.jsonl \
  --idx 8 \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --version v1.0-trainval \
  --out_png /data/patrick/packed_nuscenes_v1/preview_from_npz_idx8.png

## Quantitative Check
python scripts/sanity_npz_shapes.py \
  --manifest /data/patrick/packed_nuscenes_v1/manifest.jsonl \
  --idx 8
python scripts/sanity_npz_shapes.py \
  --manifest /data/patrick/packed_nuscenes_v2_parallel/manifest.jsonl \
  --idx 8
python scripts/sanity_npz_shapes.py \
  --manifest /data/patrick/packed_nuscenes_v1/manifest.jsonl \
  --idx 20345
python scripts/sanity_npz_shapes.py \
  --manifest /data/patrick/packed_nuscenes_v2_parallel/manifest.jsonl \
  --idx 20345
python scripts/sanity_npz_shapes.py \
  --manifest /data/patrick/packed_nuscenes_v1/manifest.jsonl \
  --random
python scripts/sanity_npz_shapes.py \
  --manifest /data/patrick/packed_nuscenes_v2_parallel/manifest.jsonl \
  --random