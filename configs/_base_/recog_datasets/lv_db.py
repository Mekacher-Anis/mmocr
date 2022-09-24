dataset_type = 'OCRDataset'
data_root = 'data/lv'
img_prefix = f'{data_root}'

train = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=f'{data_root}/train_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        file_storage_backend='disk',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=f'{data_root}/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        file_storage_backend='disk',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train_list = [train]

test_list = [test]