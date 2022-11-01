file_client_args = dict(backend='disk')

dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../../dicts/english_digits_extended_symbols.txt',
    with_padding=True,
    with_unknown=False,
    same_start_end=False,
    with_start=True,
    with_end=True,
    start_token='[B]',
    end_token='[E]',
    padding_token='[P]',
)

model = dict(
    type='PARSeq',
    backbone=None,
    encoder=dict(type='PARSeqEncoder'),
    decoder=dict(
        type='PARSeqDecoder',
        postprocessor=dict(type='AttentionPostprocessor'),
        module_loss=dict(
            type='CEModuleLoss', reduction='mean', ignore_first_char=True, flatten=True),
        dictionary=dictionary),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ignore_empty=True,
        min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(128, 32), keep_ratio=False),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(type='TorchVisionWrapper', op='RandAugment', num_ops=3, magnitude=5)
        ]),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(type='TorchVisionWrapper', op='RandomInvert')
        ]),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='GaussianBlur',
                kernel_size=(5, 9)
            )
        ]),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(
                type='ImgAugWrapper',
                args=[dict(cls='AdditivePoissonNoise', scale=(0, 40), per_channel=True)]
            )
        ]),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(128, 32), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]