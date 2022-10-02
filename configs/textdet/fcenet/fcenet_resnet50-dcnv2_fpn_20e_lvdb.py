_base_ = [
    '_base_fcenet_resnet50-dcnv2_fpn.py',
    '../_base_/datasets/lvdb.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_20e.py',
]

file_client_args = dict(backend='disk')

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
# learning policy
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=1500),
]