##########-------------- MixFormerV2-Base-----------------##########

### LaSOT test and evaluation
python tracking/test.py mixformer2_vit_online 288_depth8_score \
 --dataset lasot --threads 32 --num_gpus 8 \
 --params__model models/mixformer2_base_b_online.pth.tar \
 --params__search_area_scale 5.0 \
 --debug 0
# python tracking/analysis_results.py --dataset_name lasot --tracker_param 288_depth8_score
