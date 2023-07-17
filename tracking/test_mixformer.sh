##########-------------- MixFormerV2-Base-----------------##########

### LaSOT test and evaluation
python tracking/test.py mixformer2_vit_online 288_depth8_score \
 --dataset lasot --threads 32 --num_gpus 8 \
 --params__model models/mixformerv2_base.pth.tar \
 --params__search_area_scale 5.0 \
 --debug 0
# python tracking/analysis_results.py --dataset_name lasot --tracker_param 288_depth8_score

### TrackingNet test and pack
# python tracking/test.py mixformer2_vit_online 288_depth8_score \
#  --dataset trackingnet --threads 32 --num_gpus 8 \
#  --params__model models/mixformerv2_base.pth.tar \
#  --params__search_area_scale 5.0 \
#  --debug 0
# python lib/test/utils/transform_trackingnet.py --tracker_name mixformer2_vit_online --cfg_name 288_depth8_score

### GOT10k test and pack
# python tracking/test.py mixformer2_vit_online 288_depth8_score \
#  --dataset got10k_test --threads 32 --num_gpus 8 \
#  --params__model models/mixformerv2_base.pth.tar \
#  --params__search_area_scale 5.0 \
#  --debug 0
# python lib/test/utils/transform_got10k.py --tracker_name mixformer2_vit_online --cfg_name 288_depth8_score



##########-------------- MixFormerV2-Small-----------------##########

### LaSOT test and evaluation
# python tracking/test.py mixformer2_vit_online 224_depth4_mlp1_score \
#  --dataset lasot --threads 32 --num_gpus 8 \
#  --params__model models/mixformerv2_small.pth.tar \
#  --params__search_area_scale 4.4 \
#  --debug 0
# python tracking/analysis_results.py --dataset_name lasot --tracker_param 224_depth4_mlp1_score

### TrackingNet test and pack
# python tracking/test.py mixformer2_vit_online 224_depth4_mlp1_score \
#  --dataset trackingnet --threads 32 --num_gpus 8 \
#  --params__model models/mixformerv2_small.pth.tar \
#  --params__search_area_scale 4.55 \
#  --debug 0
# python lib/test/utils/transform_trackingnet.py --tracker_name mixformer2_vit_online --cfg_name 224_depth4_mlp1_score

### GOT10k test and pack
# python tracking/test.py mixformer2_vit_online 224_depth4_mlp1_score \
#  --dataset got10k_test --threads 32 --num_gpus 8 \
#  --params__model models/mixformerv2_small.pth.tar \
#  --params__search_area_scale 4.55 \
#  --debug 0
# python lib/test/utils/transform_got10k.py --tracker_name mixformer2_vit_online --cfg_name 224_depth4_mlp1_score

