dataset_type = 'OCRDataset'
data_root = 'data/lv'

lvdb_rec_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='recog_train_label.json',
    test_mode=False,
    pipeline=None)

lvdb_rec_val = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='recog_val_label.json',
    test_mode=False,
    pipeline=None)

lvdb_rec_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='recog_test_label.json',
    test_mode=False,
    pipeline=None)