_base_ = [
    '../_base_/datasets/lvdb.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_master_resnet31.py',
]

optim_wrapper = dict(optimizer=dict(lr=4e-4))
# learning policy
param_scheduler = [
    dict(type='LinearLR', end=100, by_epoch=False),
    dict(type='MultiStepLR', milestones=[4], end=5),
]

train_dataset = dict(
    type='ConcatDataset', datasets=[_base_.lvdb_rec_train], pipeline=_base_.train_pipeline)
val_dataset = dict(
    type='ConcatDataset', datasets=[_base_.lvdb_rec_val], pipeline=_base_.test_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=[_base_.lvdb_rec_test], pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=300,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_evaluator = dict(
    dataset_prefixes=['LVDB'])
test_evaluator = val_evaluator

load_from='/home/mekachera/bachelorarbeit/mmocr/work_dirs/master_resnet31_5e_lvdb/epoch_3.pth'