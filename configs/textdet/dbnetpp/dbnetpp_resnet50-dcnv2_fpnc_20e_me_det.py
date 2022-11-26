_base_ = [
    '_base_dbnetpp_resnet50-dcnv2_fpnc.py',
    '../_base_/datasets/me_det.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_20e.py',
]

# dataset settings
me_det_train = _base_.me_det_train
me_det_train.pipeline = _base_.train_pipeline
me_det_val = _base_.me_det_test
me_det_val.pipeline = _base_.test_pipeline
me_det_test = me_det_val

train_dataloader = dict(
    batch_size=23,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=me_det_train)

val_dataloader = dict(
    batch_size=5,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=me_det_val)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)

load_from = 'https://sf.anismk.de/static/textdet/dbnetpp_resnet50-dcnv2_fpnc_20e_lvdb_epoch_20.pth'