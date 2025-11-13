Dinov3

match

-match.py:
-match_pca.py:

Usage:

```
python match.py \
  --ref_img /home/yuan/codegen/Codegen/Utils/data/mujoco_output/039_mug_0.png \
  --ref_json /home/yuan/codegen/Codegen/Utils/data/mujoco_output/039_mug_0.json \
  --tgt_img /home/yuan/codegen/Codegen/Utils/data/mujoco_output/039_mug_2.png \
  --model_location /home/yuan/codegen/visual/arxiv/projects/dinov3 \
  --weight_path /home/yuan/codegen/visual/arxiv/projects/dinov3/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --out /home/yuan/codegen/Codegen/Utils/data/dinov3_output/out_correspondence.png
```
```
python match_pca.py \
  --ref_img /home/yuan/codegen/Codegen/Utils/data/mujoco_output/039_mug_0.png \
  --ref_json /home/yuan/codegen/Codegen/Utils/data/mujoco_output/039_mug_0.json \
  --tgt_img /home/yuan/codegen/Codegen/Utils/data/mujoco_output/039_mug_2.png \
  --model_location /home/yuan/codegen/visual/arxiv/projects/dinov3 \
  --weight_path /home/yuan/codegen/visual/arxiv/projects/dinov3/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --out /home/yuan/codegen/Codegen/Utils/data/dinov3_output/out_correspondence_pca.png
```
