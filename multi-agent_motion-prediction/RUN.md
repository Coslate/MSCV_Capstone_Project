
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


# Dataset Split
python scripts/make_splits.py \
  --manifest /data/patrick/packed_nuscenes_v1/manifest.jsonl \
  --out_dir /data/patrick/packed_nuscenes_v1/splits \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --seed 2025

python scripts/make_splits.py \
  --manifest /data/patrick/packed_nuscenes_v2_parallel/manifest.jsonl \
  --out_dir /data/patrick/packed_nuscenes_v2_parallel/splits \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --seed 2025

head -n 2000 /data/patrick/packed_nuscenes_v1/splits/manifest.train.jsonl > /data/patrick/packed_nuscenes_v1/splits/mini/manifest.train_2k.jsonl
head -n 500 /data/patrick/packed_nuscenes_v1/splits/manifest.val.jsonl > /data/patrick/packed_nuscenes_v1/splits/mini/manifest.val_500.jsonl

# Train on mini for testing
python -m scripts.train \
  --train_manifest /data/patrick/packed_nuscenes_v1/splits/mini/manifest.train_2k.jsonl \
  --val_manifest   /data/patrick/packed_nuscenes_v1/splits/mini/manifest.val_500.jsonl \
  --batch_size 32 \
  --epochs 1 \
  --val_every 10 \
  --lr 1e-3 \
  --no_lr_sched \
  --warmup_ratio 0.05 \
  --min_lr 1e-5 \
  --warmup_init_lr 1e-5 \
  --hidden 128 --num_layers 1 --d_model 128 \
  --device cuda \
  --amp_dtype fp16 \
  --out_dir runs/baseline_single_2k \
  --wandb --wandb_project multi-agent-motion-prediction --wandb_run_name mmp_bl_single_2k \
  --log_every 1

# Train on full dataset - baseline model GRU
python -m scripts.train \
  --train_manifest /data/patrick/packed_nuscenes_v1/splits/manifest.train.jsonl \
  --val_manifest   /data/patrick/packed_nuscenes_v1/splits/manifest.val.jsonl \
  --batch_size 32 \
  --epochs 12 \
  --val_every 200 \
  --lr 1e-3 \
  --no_lr_sched \
  --warmup_ratio 0.03 \
  --min_lr 1e-4 \
  --warmup_init_lr 1e-4 \
  --hidden 128 --num_layers 1 --d_model 128 \
  --device cuda \
  --amp_dtype fp16 \
  --out_dir runs/baseline_fulldataset_12k_nosched \
  --wandb --wandb_project multi-agent-motion-prediction --wandb_run_name mmp_bl_fulldataset_12k_lr_nosched \
  --log_every 10

python -m scripts.train \
  --train_manifest /data/patrick/packed_nuscenes_v1/splits/manifest.train.jsonl \
  --val_manifest   /data/patrick/packed_nuscenes_v1/splits/manifest.val.jsonl \
  --batch_size 32 \
  --epochs 12 \
  --val_every 200 \
  --lr 1e-3 \
  --warmup_ratio 0.03 \
  --min_lr 1e-4 \
  --warmup_init_lr 1e-4 \
  --hidden 128 --num_layers 1 --d_model 128 \
  --device cuda \
  --amp_dtype fp16 \
  --out_dir runs/mmp_bl_fulldataset_12k_lr_warmupcos_lr1e-3_wup0.03_minlr1e-4_e12 \
  --wandb --wandb_project multi-agent-motion-prediction --wandb_run_name mmp_bl_fulldataset_12k_lr_warmupcos_lr1e-3_wup0.03_minlr1e-4_e12 \
  --log_every 10

python -m scripts.train \
  --train_manifest /data/patrick/packed_nuscenes_v1/splits/manifest.train.jsonl \
  --val_manifest   /data/patrick/packed_nuscenes_v1/splits/manifest.val.jsonl \
  --batch_size 32 \
  --epochs 15 \
  --val_every 200 \
  --lr 3e-3 \
  --warmup_ratio 0.05 \
  --min_lr 3e-4 \
  --warmup_init_lr 1e-4 \
  --hidden 128 --num_layers 1 --d_model 128 \
  --device cuda \
  --amp_dtype fp16 \
  --out_dir runs/mmp_bl_fulldataset_12k_lr_warmupcos_lr3e-3_wup0.05_minlr3e-4_e15 \
  --wandb --wandb_project multi-agent-motion-prediction --wandb_run_name mmp_bl_fulldataset_12k_lr_warmupcos_lr3e-3_wup0.05_minlr3e-4_e15 \
  --log_every 10


# Evaluation baseline (not run yet)
## Use FDE rank and select top worse 30 to save CSV+visualization for analyze (single rollout)
python -m scripts.eval \
  --manifest /data/.../manifest.test.jsonl \
  --ckpt runs/.../best.pt \
  --batch_size 64 --device cuda \
  --rank_by FDE \
  --topk_vis 30 --save_vis_dir runs/.../vis_top30_fde \
  --dump_csv runs/.../test_per_agent.csv \
  --vis_with_map

## Use FDE rank and select top worse 50 to save CSV+visualization for analyze (multi rollout)
python -m scripts.eval \
  --manifest /data/.../manifest.test.jsonl \
  --ckpt runs/.../best.pt \
  --multirollout --K 6 \
  --rank_by minADE_K \
  --topk_vis 50 --save_vis_dir runs/.../vis_top50_minADEK \
  --dump_csv runs/.../test_per_agent.csv

## Simple visualization, no selecting top worse, draw single idx with all agents in it.
python -m scripts.visualize \
  --manifest /data/.../manifest.val.jsonl \
  --ckpt runs/.../best.pt \
  --device cuda \
  --out_dir runs/.../viz_single \
  single --idx 1234 --include_map

## Simple visualization, no selecting top worse, draw single idx with single agents in it.
python -m scripts.visualize \
  --manifest /data/.../manifest.val.jsonl \
  --ckpt runs/.../best.pt \
  --device cuda \
  --out_dir runs/.../viz_single_agent7 \
  single --idx 1234 --agent 7 --include_map

## Simple visualization, no selecting top worse, draw sequence idx with single agents in it.
python -m scripts.visualize \
  --manifest /data/.../manifest.val.jsonl \
  --ckpt runs/.../best.pt \
  --device cuda \
  --out_dir runs/.../viz_sequence_sceneX \
  sequence --start_idx 5000 --num 120 --ensure_same_scene --include_map --fps 10 --make_video