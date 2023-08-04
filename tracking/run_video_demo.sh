# We only support manually setting the bounding box of first frame and save the results in debug directory.

##########-------------- MixFormerV2-Base-----------------##########
python tracking/video_demo.py \
  mixformer2_vit_online \
  288_depth8_score \
  xxx.mp4  \
  --optional_box [YOUR_X] [YOUR_Y] [YOUR_W] [YOUR_H] \ 
  --params__model models/mixformerv2_base.pth.tar --debug 1 \
  --params__search_area_scale 4.5 --params__update_interval 25 --params__online_size 1
