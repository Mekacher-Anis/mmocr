_base_ = [
    '../_base_/datasets/lvdb.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_satrn_shallow.py',
]

# dataset settings
train_list = [ _base_.lvdb_rec_train ]
val_list = [ _base_.lvdb_rec_val ]
test_list = [ _base_.lvdb_rec_test ]

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
val_dataset = dict(
    type='ConcatDataset', datasets=val_list, pipeline=_base_.test_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=3e-4))

train_dataloader = dict(
    batch_size=100,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=50,
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
    dataset_prefixes=[ 'LVDB' ])
test_evaluator = val_evaluator

load_from = '/home/mekachera/bachelorarbeit/mmocr/work_dirs/satrn_shallow_5e_st_mj/epoch_5.pth'