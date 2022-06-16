_base_ = '../sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
