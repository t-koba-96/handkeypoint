# model settings
model = dict(
    type="TTFNet",
    pretrained="modelzoo://resnet34",
    backbone=dict(
        type="ResNet",
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=False,
        style="pytorch"),
    neck=None,
    bbox_head=dict(
        type="TTFHead",
        inplanes=(64, 128, 256, 512),
        head_conv=128,
        wh_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=2,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        shortcut_cfg=(1, 2, 3),
        norm_cfg=dict(type="BN"),
        alpha=0.54,
        hm_weight=1.,
        wh_weight=5.))
cudnn_benchmark = True

# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=100)

# dataset settings
dataset_type = "CustomDataset"
data_root = "./data/EgoHands/"
img_norm_cfg = dict(
    mean=[121.02893939177196,112.60815201123556,103.15652412970861], 
    std=[61.08331256843366,57.68156265485914,59.75120580722921], 
    to_rgb=True)

# all data are used for training, validation and testing
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file="./data/EgoHands/annotations/annotations.pkl",
        img_prefix="",
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
    
    # data augmentation
        flip_ratio=0.5,
        extra_aug=dict(
            photo_metric_distortion=dict(
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            random_crop=dict(
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
        with_mask=False,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False),
    
    val=dict(
        type=dataset_type,
        ann_file="./data/EgoHands/annotations/annotations.pkl",
        img_prefix="",
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False),
        
    test=dict(
        type=dataset_type,
        ann_file="./data/EgoHands/annotations/annotations.pkl",
        img_prefix="",
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))


# optimizer
optimizer = dict(type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0004, 
        paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    step=[18, 22])

# save the model after each epoch
checkpoint_config = dict(interval=1)
bbox_head_hist_config = dict(
    model_type=["ConvModule", "DeformConvPack"],
    sub_modules=["bbox_head"],
    save_every_n_steps=500)
log_config = dict(
     interval=50,
     hooks=[
         dict(type="TextLoggerHook"),
    ])

# yapf:enable
# runtime settings
total_epochs = 24
log_level = "INFO"
work_dir = "work_dirs/ttfnet"
load_from = None
resume_from = None
workflow = [("train", 1)]
