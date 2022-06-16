# dataset settings
dataset_type = 'CocoDataset'
# classes = ("solar_S","solar_M","solar_B")
# data_root = '/media/ymhj/data/kxp/kd/mdistiller-master/detection/datasets/coco/'
# data_root = '/media/ymhj/data/kxp/RCD/coco/'
data_root = '/media/ymhj/data/gxy/yolov5-6.0/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(2048,2048), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048,2048),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=20,
    train=dict(
        type=dataset_type,
        # classes=("solar_S","solar_M","solar_B"),
        # ann_file=data_root + 'annotations/train_cityscapes.json',
        # img_prefix=data_root + 'train_cityscapes/images/',
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # classes=("solar_S","solar_M","solar_B"),
        # ann_file=data_root + 'annotations/val_cityscapes.json',
        # img_prefix=data_root + 'val_cityscapes/images/',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # classes=("solar_S","solar_M","solar_B"),
    
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=5, metric='bbox')
