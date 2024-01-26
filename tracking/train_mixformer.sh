# There are the detailed training settings for MixFormerV2-b and MixFormerV2-s.


### Stage1 Dense-to-Sparse Distillation
# 1. download pretrained MAE models (mae_pretrain_vit_base.pth/mae_pretrain_vit_large.pth) at https://github.com/facebookresearch/mae
# 2. set the proper pretrained vit models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/mixformer2_vit/CONFIG_NAME.yaml.
# 3. download pretrained MixFormer teacher models (mixformer_vit_base_online.pth.tar/mixformer_vit_large_online.pth.tar) at https://github.com/MCG-NJU/MixFormer
# 4. uncomment the following code to train corresponding trackers.

python tracking/train.py --script mixformer2_vit \
 --config student_288_depth12 \
 --save_dir . \
 --distill 1 --script_teacher mixformer_vit --config_teacher teacher_mixvit_b \
 --checkpoint_teacher_path ./models/mixformer_vit_base_online.pth.tar \
 --mode multiple --nproc_per_node 8


### Stage2 Deep-to-Shallow Distillation
# with trained student model checkpoint from stage1

## 2.1 Remove layers
# python tracking/train.py --script mixformer2_vit_stu \
#  --config student_288_depth12to8 \
#  --save_dir . \
#  --distill 1 --script_teacher mixformer2_vit --config_teacher teacher_288_depth12 \
#  --checkpoint_teacher_path PATH/TO/STAGE1/CHECKPOINT \
#  --mode multiple --nproc_per_node 8


## 2.2 Continuing finetune
# python tracking/train.py --script mixformer2_vit_stu \
#  --config student_288_depth8 \
#  --save_dir . \
#  --distill 1 --script_teacher mixformer2_vit --config_teacher teacher_288_depth12 \
#  --checkpoint_teacher_path PATH/TO/REMOVE/CHECKPOINT \
#  --mode multiple --nproc_per_node 8


## Train Online Head, which is similar to mixformer
# python tracking/train.py --script mixformer2_vit_online \
#  --config 288_depth8_score \
#  --save_dir . \
#  --mode multiple --nproc_per_node 1 \
#  --static_model PATH/TO/TRAINED/MODEL
