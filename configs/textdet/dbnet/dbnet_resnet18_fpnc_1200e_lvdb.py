_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
    '../_base_/datasets/lvdb.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# dataset settings
lvdb_det_train = _base_.lvdb_det_train
lvdb_det_train.pipeline = _base_.train_pipeline
lvdb_det_val = _base_.lvdb_det_val
lvdb_det_val.pipeline = _base_.test_pipeline
lvdb_det_test = _base_.lvdb_det_test
lvdb_det_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=lvdb_det_train)

val_dataloader = dict(
    batch_size=10,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=lvdb_det_val)

test_dataloader = dict(
    batch_size=10,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=lvdb_det_test)

auto_scale_lr = dict(base_batch_size=16)

load_from = '/home/anis/.cache/torch/hub/checkpoints/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
