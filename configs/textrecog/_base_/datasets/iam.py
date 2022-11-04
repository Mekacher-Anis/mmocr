iam_rec_train = dict(
    type='OCRDataset',
    data_root='data/rec/iam/',
    data_prefix=dict(img_path='words/'),
    ann_file='iam_word_recog_ann.json',
    test_mode=False,
    pipeline=None)