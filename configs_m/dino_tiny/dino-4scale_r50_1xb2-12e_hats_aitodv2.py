_base_ = [
    './dino-4scale_r50_1xb2-12e_aitodv2.py'
]

model = dict(
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='HatsCost', iou_mode='hausdorff', weight=2.0,
                     iou_kwargs=dict(lambda1=2.5,
                                     lambda3=7,
                                     using_central=True,
                                     pow_value=4.0,
                                     slope=2.0,
                                     bias=30))
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR
