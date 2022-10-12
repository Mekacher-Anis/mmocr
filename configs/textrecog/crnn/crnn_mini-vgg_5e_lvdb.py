# training schedule for 1x
_base_ = [
    '../_base_/datasets/lvdb.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_crnn_mini-vgg.py',
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=50), )

train_dataloader = dict(
    batch_size=64,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[ _base_.lvdb_rec_train ],
        pipeline=_base_.train_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[ _base_.lvdb_rec_test ],
        pipeline=_base_.test_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[ _base_.lvdb_rec_val ],
        pipeline=_base_.test_pipeline))

val_evaluator = dict(
    dataset_prefixes=['LVDB'])
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64 * 4)
