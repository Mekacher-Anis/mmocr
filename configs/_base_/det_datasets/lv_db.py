dataset_type = 'IcdarDataset'
data_root = 'data/lv'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_train.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

train_list = [train]

test_list = [test]