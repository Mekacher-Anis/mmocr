dataset_type = 'IcdarDataset'
lvdb_data_root = 'data/lv'

lvdb_det_train = dict(
    type=dataset_type,
    data_root=lvdb_data_root,
    ann_file='instances_train.json',
    data_prefix=dict(img_path='imgs/'),
    pipeline=None
)

lvdb_det_val = dict(
    type=dataset_type,
    data_root=lvdb_data_root,
    ann_file='instances_val.json',
    data_prefix=dict(img_path='imgs/'),
    pipeline=None
)

lvdb_det_test = dict(
    type=dataset_type,
    data_root=lvdb_data_root,
    ann_file='instances_test_subset.json',
    data_prefix=dict(img_path='imgs/'),
    pipeline=None
)