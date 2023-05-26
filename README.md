# MixFormerV2
The official implementation of paper: [**MixFormerV2: Efficient Fully Transformer Tracking**](https://arxiv.org/abs/2305.15896).

## Model Framework
![model](tracking/model.png)

## Distillation Training Pipeline
![training](tracking/training.png)


## News

- **[May 26, 2023]** Code is available now!


## Install the environment
Use the Anaconda
``` bash
conda create -n mixformer2 python=3.6
conda activate mixformer2
bash install_requirements.sh
```


## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${MixFormerV2_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- train2017
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train MixFormerV2

Training with multiple GPUs using DDP. More details of other training settings can be found at `tracking/train_mixformer.sh`.

``` bash
bash tracking/train_mixformer.sh
```

## Test and evaluate MixFormerV2 on benchmarks
- LaSOT/GOT10k-test/TrackingNet/OTB100/UAV123/TNL2k. More details of test settings can be found at `tracking/test_mixformer.sh`.

``` bash
bash tracking/test_mixformer.sh
```

## Contant
Tianhui Song: 191098194@smail.nju.edu.cn

Yutao Cui: cuiyutao@smail.nju.edu.cn 


## Citiation
@misc{mixformerv2,
      title={MixFormerV2: Efficient Fully Transformer Tracking}, 
      author={Yutao Cui and Tianhui Song and Gangshan Wu and Limin Wang},
      year={2023},
      eprint={2305.15896},
      archivePrefix={arXiv}
}
