# MMSegmentation
> 训练我们自己的数据集

首先标签图像是单通道的灰度图，也就是```mode="P"```, 每一类用一个像素值表示，0，1，2，3...，最后一类是背景。

[generate_anns.py](./datasets/generate_anns.py)

train和test文件夹中的图片名称相同，后缀不同，图像为jpg，标签为png。

文件夹结构：

```Bash
|——data
|   |——my_dataset
|   |   |——img_dir
|   |   |   |——train
|   |   |   |   |——1.jpg
|   |   |   |   |——2.jpg
|   |   |   |——test
|   |   |
|   |   |——ann_dir
|   |   |   |——train
|   |   |   |   |——1.png
|   |   |   |   |——2.png
|   |   |   |——test
```

## 1. ./mmseg/datasets/my_dataset.py

在`./mmseg/datasets/`下定义自己的数据集。

```python
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyDataset(CustomDataset):
    # 写你实际的类别名就好了，最后再加上一个background
    CLASSES = (
        'object', 'background'
    )
    # 这个数量与上面个数对应就好了

    PALETTE = [[120, 120, 120], [180, 120, 120]]

    def __init__(self, **kwargs):
        super(MyDataset, self).__init__(
            **kwargs
        )
```

并在`./mmseg/datasets/__init__.py`中导入

```python
from .my_dataset import MyDataset

__all__ = [

    'MyDataset'
]
```

## 2. 编写或改动config文件

```python
# 参考 pspnet_r50-d8_512x512_160k_ade20k 来做的

# 1.model：根据'../_base_/models/pspnet_r50-d8.py'
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=2,   # 注意改这个类别数,我的是27类+最后一个background
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,       # 注意改这个值，
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# 2. data：根据'../_base_/datasets/ade20k.py'
# dataset settings
dataset_type = 'MyDataset'   # 自己mmseg/datasets/my_dataset.py中的类名
# data_root = 'data/my_dataset'
data_root = '/user34/fuhao/datasets/urpc2020/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(1280, 1024), ratio_range=(0.5, 2.0)),   # img_scale我还是就用的我原图的大小，你可以改成你的大小，其他地方同步改
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,   # 显存不够，来把这改小
    workers_per_gpu=2,    
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='ann/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='ann/test',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='ann/test',
        pipeline=test_pipeline))


# 3. 根据：'../_base_/default_runtime.py'
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True


# 4. schedules：根据'../_base_/schedules/schedule_160k.py'
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)   # 根据自己情况改学习率吧
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)

```