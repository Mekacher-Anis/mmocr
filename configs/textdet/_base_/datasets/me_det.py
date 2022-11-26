dataset_type = 'IcdarDataset'
lvdb_data_root = 'data/medet/equation_detection'

me_det_train = dict(
    type=dataset_type,
    data_root=lvdb_data_root,
    ann_file='me_det_train.json',
    data_prefix=dict(img_path='./'),
    pipeline=None
)

me_det_test = dict(
    type=dataset_type,
    data_root=lvdb_data_root,
    ann_file='me_det_test.json',
    data_prefix=dict(img_path='./'),
    pipeline=None
)