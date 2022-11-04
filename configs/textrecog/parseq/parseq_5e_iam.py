_base_ = [
    '../_base_/datasets/iam.py',
    '../_base_/datasets/lvdb.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_parseq_mha.py',
]

train_dataset = dict(
    type='ConcatDataset', datasets=[_base_.iam_rec_train], pipeline=_base_.train_pipeline)
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
