# Traffic-Rule-Aware Motion Prediction — Minimal (nuScenes)
- 統一輸出介面（images / lidar / calib / ego_history / agents_history / map / timestamps）
- nuScenes v1.0 loader（含多代理歷史、ego 歷史、地圖切片 token）
- DataModule + 簡易 BEV 視覺化 + demo 腳本

## 快速開始
pip install -r requirements.txt
python scripts/list_nuscenes_sample_tokens.py --dataroot /path/to/nuscenes --n 5
python scripts/demo_nuscenes.py --dataroot /path/to/nuscenes --sample_token <貼這裡> --out_png /tmp/preview.png
