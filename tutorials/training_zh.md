# 训练教程

本方法使用多阶段训练，本文档通过将 MixFormer-ViT-Base 蒸馏为一个8层模型为示例，说明训练的流程。

## 一阶段训练

一阶段为 **Dense-to-Sparse** 蒸馏，用来引入 Prediction Token，替换定位头。
我们使用 MixFormer-ViT-Base 为教师模型，下载好 [checkpoint](https://github.com/MCG-NJU/MixFormer)（假设权重保存在 `models/mixformer_vit_base_online.pth.tar`），教师模型使用的配置文件为 `experiments/mixformer_vit/teacher_mixvit_b.yaml`。
学生模型使用的配置文件为 `experiments/mixformer2_vit/student_288_depth12.yaml`，主要的数据，训练等超参也在学生配置文件中定义。

执行训练：
```bash
python tracking/train.py \
 --script mixformer2_vit \
 --config student_288_depth12 \
 --save_dir . \
 --distill 1 \
 --script_teacher mixformer_vit \
 --config_teacher teacher_mixvit_b \
 --checkpoint_teacher_path ./models/mixformer_vit_base_online.pth.tar \
 --mode multiple --nproc_per_node 8
```

说明：
- `--script`/`--teacher_script` 参数指示了模型结构，这里教师模型使用原始 `mixformer_vit`，学生为本文提出的模型结构 `mixformer2_vit`。
- `--config`/`--teacher_config` 参数指示模型所使用的配置文件。
- `--checkpoint_teacher_path` 为教师模型权重文件路径。
- 第一阶段训练完成后，会得到12层的 mixformer2_vit 的模型 checkpoint，应该会保存在 `checkpoints/train/mixformer2_vit/student_288_depth12/` 目录下。

## 二阶段训练
第二阶段为 **Deep-to-Shallow** 蒸馏，用来撤除模型中的部分层，减小模型层数。

### Remove 过程
首先是remove过程，在这个阶段我们会在训练中逐渐删除模型中的部分层。

执行训练：
```bash
python tracking/train.py \
 --script mixformer2_vit_stu \
 --config student_288_depth12to8 \
 --save_dir . \
 --distill 1 \
 --script_teacher mixformer2_vit \
 --config_teacher teacher_288_depth12 \
 --checkpoint_teacher_path ./checkpoints/train/mixformer2_vit/student_288_depth12/MixFormer_ep0500.pth.tar \
 --mode multiple --nproc_per_node 8
```

说明：
- 我们使用上一阶段得到的checkpoint作为新的教师模型，模型配置文件为 `experiments/mixformer2_vit/teacher_288_depth12.yaml`。
  - 该权重也会用来初始化学生模型，需要在学生模型的配置文件中设置 `MODEL.BACKBONE.PRETRAINED_PATH` 为该权重文件路径。
- 学生模型 script 为 `mixformer2_vit_stu`，其与 `mixformer2_vit` 的结构定义并无区别，只是代码中加入了一些remove的逻辑。
- 学生模型的配置文件为 `experiments/mixformer2_vit_stu/student_288_depth12to8.yaml`，定义了一些remove过程的关键参数。
  - remove 过程只训练 40 epoch，即二阶段的前 40 epoch。
  - `TRAIN.REMOVE_LAYERS` 指定了需要撤除层的索引。
  - `TRAIN.DISTILL_LAYERS_[STUDENT|TEACHER]` 指定指定了学生和教师模型进行特征监督的对应层的索引。
- 训练完成后，会得到已经删除部分层的模型 checkpoint，应该会保存在 `checkpoints/train/mixformer2_vit_stu/student_288_depth12to8/` 路径下，注意该权重依然是12层的，只是 `REMOVE_LAYERS` 中的层已经没用。


### 继续 Finetune
至此，已经完成了模型backbone的压缩，我们加载剩余的层继续finetune。

执行训练：
```bash
python tracking/train.py \
 --script mixformer2_vit_stu \
 --config student_288_depth8 \
 --save_dir . \
 --distill 1 \
 --script_teacher mixformer2_vit \
 --config_teacher teacher_288_depth12 \
 --checkpoint_teacher_path ./checkpoints/train/mixformer2_vit/student_288_depth12/MixFormer_ep0500.pth.tar \
 --mode multiple --nproc_per_node 8
```

说明：
- 教师模型依然和前面一样。
- 学生模型的配置文件为 `experiments/mixformer2_vit_stu/student_288_depth8.yaml`，加载上一步骤中保存的 checkpoint。
  - 在配置文件中设置 `MODEL.BACKBONE.PRETRAINED_PATH` 为前一步 checkpoint 路径。
  - 注意该模型已经是8层的模型，在加载模型时需要进行 state_dict 中的一些 key 的修改，去掉不需要的层（在配置中的 `TRAIN.INVALID_LAYERS` 指示），reset 剩余的层的索引。代码中已实现。
  - 特征监督所对应的索引也要相应修改。
- 训练完成后，就得到了最终的压缩后的8层MixFormerV2模型，应该保存在 `checkpoints/train/mixformer2_vit_stu/student_288_depth8/` 路径下。


## Online 训练
最后，和 MixFormerV1 一样，我们额外训练一个 online 得分模块，来实现推理时模版更新。
执行训练：
```bash
python tracking/train.py --script mixformer2_vit_online \
 --config 288_depth8_score \
 --save_dir . \
 --mode multiple --nproc_per_node 1 \
 --static_model ./checkpoints/train/mixformer2_vit_stu/student_288_depth8/MixFormer_ep0500.pth.tar
```
说明：
- `--static_model` 即为以上训练完成的模型，在这个阶段不会更新参数，只会额外训练一个MLP得分模块。
- 最终模型应该保存在 `checkpoints/train/mixformer2_vit_online/288_depth8_score/` 路径下。
