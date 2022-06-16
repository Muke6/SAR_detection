_base_ = '../cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_20e_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
