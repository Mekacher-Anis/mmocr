ic13_rec_data_root = 'data/rec/icdar2013/'

ic13_rec_train = dict(
    type='OCRDataset',
    data_root=ic13_rec_data_root,
    ann_file='train_label.json',
    data_prefix=dict(img_path='crops/train/'),
    test_mode=False,
    pipeline=None)

ic13_rec_test = dict(
    type='OCRDataset',
    data_root=ic13_rec_data_root,
    ann_file='test_label.json',
    data_prefix=dict(img_path='crops/test/'),
    test_mode=True,
    pipeline=None)
