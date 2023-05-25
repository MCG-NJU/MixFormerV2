# There are the detailed training settings for MixFormerV2-b and MixFormerV2-s.
# 1. download pretrained MAE models (mae_pretrain_vit_base.pth/mae_pretrain_vit_large.pth) at https://github.com/facebookresearch/mae
# 2. set the proper pretrained vit models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/mixformer2_vit/CONFIG_NAME.yaml.
# 3. download pretrained MixFormer teacher models (mixformer_vit_base_online.pth.tar/mixformer_vit_large_online.pth.tar) at https://github.com/MCG-NJU/MixFormer
# 4. place or symbal link the checkpoint at path SAVE_DIR/checkpoints/train/mixformer_vit/TEACHER_CONFIG_NAME
# 5. uncomment the following code to train corresponding trackers.

### Stage1 Dense-to-Sparse Distillation
python tracking/train.py --script mixformer2_vit \
 --config student_288_depth12 \
 --save_dir . \
 --distill 1 --script_teacher mixformer_vit --config_teacher teacher_mixvit_b \
 --mode multiple --nproc_per_node 1