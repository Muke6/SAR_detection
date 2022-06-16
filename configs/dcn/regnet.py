_base_ = '../regnet/cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
