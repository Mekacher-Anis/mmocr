_base_ = [
    '../_base_/datasets/iam.py',
    '../_base_/datasets/lvdb.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_10k.py',
    '_base_parseq_mha.py',
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=501,
        by_epoch=False,
        save_best=['LVDB/recog/1-N.E.D_exact', 'LVDB/recog/word_acc'],
        rule='greater'
    ), 
)

# custom pipeline to add color jitter
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        ignore_empty=True,
        min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(128, 32), keep_ratio=False),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(type='TorchVisionWrapper', op='RandAugment', num_ops=3, magnitude=5)
        ]),
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(type='TorchVisionWrapper', op='RandomInvert')
        ]),
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='GaussianBlur',
                kernel_size=(5, 9)
            )
        ]),
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(
                type='ImgAugWrapper',
                args=[dict(cls='AdditivePoissonNoise', lam=(0, 10), per_channel=True)]
            )
        ]),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

train_dataset = dict(
    type='ConcatDataset', datasets=[_base_.iam_rec_train], pipeline=train_pipeline)
val_dataset = dict(
    type='ConcatDataset', datasets=[_base_.lvdb_rec_val], pipeline=_base_.test_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=[_base_.lvdb_rec_test], pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=100,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=300,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

test_dataloader = dict(
    batch_size=1500,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_evaluator = dict(
    dataset_prefixes=['LVDB'])
test_evaluator = val_evaluator

load_from = 'https://sf.anismk.de/static/textrecog/parseq_5e_ldvb_epoch_2.ckpt'
