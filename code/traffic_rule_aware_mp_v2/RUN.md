
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
python scripts/demo_nuscenes.py \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --sample_token e93e98b63d3b40209056d129dc53ceee \
  --out_png /tmp/preview.png

python scripts/demo_nuscenes.py \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --sample_token 1e19d0a5189b46f4b62aa47508f2983e \
  --out_png ./tmp/preview.png

python scripts/demo_nuscenes_v3.py \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --sample_token 1e19d0a5189b46f4b62aa47508f2983e \
  --past_sec 3 \
  --map_radius_m 80 \
  --with_lidar \
  --out_png ./tmp/preview_v3.png

python scripts/demo_nuscenes_v4.py \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --sample_token 1e19d0a5189b46f4b62aa47508f2983e \
  --past_sec 4 \
  --with_lidar \
  --map_radius_m 80 \
  --out_png ./tmp/preview_v4.png

python scripts/demo_nuscenes_v4.py \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --sample_token 1e19d0a5189b46f4b62aa47508f2983e \
  --past_sec 3 \
  --with_lidar \
  --map_radius_m 80 \
  --out_png ./tmp/preview_v7.png  

python scripts/demo_nuscenes_v4.py \
  --dataroot /data/Datasets/NuScenes-QA/data/nuScenes \
  --sample_token 1e19d0a5189b46f4b62aa47508f2983e \
  --past_sec 3 \
  --with_lidar \
  --map_radius_m 80 \
  --out_png ./tmp/preview_v8.png  