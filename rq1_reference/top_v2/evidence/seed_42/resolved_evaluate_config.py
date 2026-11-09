algorithm_keys = (
    'dataset',
    'frame_dir',
    'filename_tmpl',
    'img_shape',
    'sample_idx',
    'video_id',
    'type',
    'start_index',
    'total_frames',
    'target',
    'abnormal_start_frame',
    'accident_frame',
    'frame_inds',
    'clip_len',
    'num_clips',
    'frame_interval',
    'fps',
    'is_val',
    'is_test',
)
cap = dict(
    ann_file='cap_text_annotations.xls',
    data_root='data/MM-AU/CAP-DATA',
    filename_tmpl='{:06}.jpg',
    start_index=1)
clip_len = 5
custom_hooks = [
    dict(type='EpochHook'),
    dict(deadline=1791614800.1882944, seed=42, type='ReproductionStateHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'accident_anticipation',
        'kaggle_adapter',
        'amp_adapter',
    ])
d2city = dict(ann_file='annotations.csv', data_root='data/D_square-City')
dada = dict(
    ann_file='dada_text_annotations.xlsx',
    data_root='data/MM-AU/DADA-DATA',
    filename_tmpl='{:04}.png',
    start_index=1)
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,
        rule='greater',
        save_best='mAUC@',
        type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = '/workspace/top_v2_v6matched_fixed_3seeds/seed_42/run/resume_latest.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
modality = 'rgb'
model = dict(
    backbone=dict(
        conv1_kernel=[
            1,
            7,
            7,
        ],
        conv1_stride_t=1,
        depth=50,
        inflate=[
            0,
            0,
            1,
            1,
        ],
        lateral=False,
        norm_eval=False,
        pool1_stride_t=1,
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        type='ResNet3dSlowOnly'),
    cls_head=dict(
        clip_len=5,
        label_with='annotation',
        num_clips=30,
        pos_weight=10,
        two_stream=False,
        type='AnticipationHead',
        with_decoder=True,
        with_rnn=False),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    test_cfg=dict(max_testing_views=10),
    train_cfg=None,
    type='Recognizer3D')
model_wrapper_cfg = dict(
    find_unused_parameters=False, type='MMDistributedDataParallel')
nexar = dict(
    ann_file='annotations.csv',
    data_root='data/nexar-collision-prediction',
    filename_tmpl='{:06}.jpg',
    start_index=0)
num_clips = 30
optim_wrapper = dict(
    accumulative_counts=8,
    clip_grad=dict(max_norm=40, norm_type=2),
    dtype='float16',
    loss_scale=dict(growth_interval=2000, init_scale=1024.0),
    optimizer=dict(lr=0.002, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=50,
        gamma=0.1,
        milestones=[
            20,
            40,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=42)
resume = True
test_cfg = dict(fp16=False, type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        modality='rgb',
        nexar=dict(
            ann_file='annotations.csv',
            data_root=
            '/workspace/top_v2_v6matched_fixed_3seeds/seed_42/prepared_nexar',
            filename_tmpl='{:06}.jpg',
            start_index=0),
        pipeline_frame=[
            dict(
                clip_len=5,
                num_clips=30,
                test_mode=True,
                type='V6SampleFramesBeforeAccident'),
            dict(type='NativeJpegDecode'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(modality='rgb', type='Flow'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                algorithm_keys=(
                    'dataset',
                    'frame_dir',
                    'filename_tmpl',
                    'img_shape',
                    'sample_idx',
                    'video_id',
                    'type',
                    'start_index',
                    'total_frames',
                    'target',
                    'abnormal_start_frame',
                    'accident_frame',
                    'frame_inds',
                    'clip_len',
                    'num_clips',
                    'frame_interval',
                    'fps',
                    'is_val',
                    'is_test',
                    'text',
                    'have_accident',
                ),
                meta_keys=(),
                type='PackActionInputs'),
        ],
        pipeline_video=[
            dict(
                clip_len=5,
                num_clips=30,
                test_mode=True,
                type='V6SampleFramesBeforeAccident'),
            dict(type='NativeJpegDecode'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(modality='rgb', type='Flow'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                algorithm_keys=(
                    'dataset',
                    'frame_dir',
                    'filename_tmpl',
                    'img_shape',
                    'sample_idx',
                    'video_id',
                    'type',
                    'start_index',
                    'total_frames',
                    'target',
                    'abnormal_start_frame',
                    'accident_frame',
                    'frame_inds',
                    'clip_len',
                    'num_clips',
                    'frame_interval',
                    'fps',
                    'is_val',
                    'is_test',
                    'text',
                    'have_accident',
                ),
                meta_keys=(),
                type='PackActionInputs'),
        ],
        seed=42,
        split='test',
        test_mode=True,
        type='NexarSubsetDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='TestPredictionMetric')
test_pipeline_frame = [
    dict(
        clip_len=5,
        num_clips=30,
        test_mode=True,
        type='SampleFramesBeforeAccident'),
    dict(io_backend='disk', type='RawFrameDecode'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(modality='rgb', type='Flow'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(
        algorithm_keys=(
            'dataset',
            'frame_dir',
            'filename_tmpl',
            'img_shape',
            'sample_idx',
            'video_id',
            'type',
            'start_index',
            'total_frames',
            'target',
            'abnormal_start_frame',
            'accident_frame',
            'frame_inds',
            'clip_len',
            'num_clips',
            'frame_interval',
            'fps',
            'is_val',
            'is_test',
        ),
        meta_keys=(),
        type='PackActionInputs'),
]
test_pipeline_video = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=5,
        num_clips=30,
        test_mode=True,
        type='SampleFramesBeforeAccident'),
    dict(type='DecordDecode'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(modality='rgb', type='Flow'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(
        algorithm_keys=(
            'dataset',
            'frame_dir',
            'filename_tmpl',
            'img_shape',
            'sample_idx',
            'video_id',
            'type',
            'start_index',
            'total_frames',
            'target',
            'abnormal_start_frame',
            'accident_frame',
            'frame_inds',
            'clip_len',
            'num_clips',
            'frame_interval',
            'fps',
            'is_val',
            'is_test',
        ),
        meta_keys=(),
        type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=50, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        modality='rgb',
        nexar=dict(
            ann_file='annotations.csv',
            data_root=
            '/workspace/top_v2_v6matched_fixed_3seeds/seed_42/prepared_nexar',
            filename_tmpl='{:06}.jpg',
            start_index=0),
        pipeline_frame=[
            dict(
                clip_len=5,
                num_clips=30,
                test_mode=False,
                type='V6SampleFramesBeforeAccident'),
            dict(type='NativeJpegDecode'),
            dict(
                area_range=(
                    0.8,
                    1.0,
                ),
                aspect_ratio_range=(
                    1.3333333333333333,
                    1.7777777777777777,
                ),
                type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(modality='rgb', type='Flow'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                algorithm_keys=(
                    'dataset',
                    'frame_dir',
                    'filename_tmpl',
                    'img_shape',
                    'sample_idx',
                    'video_id',
                    'type',
                    'start_index',
                    'total_frames',
                    'target',
                    'abnormal_start_frame',
                    'accident_frame',
                    'frame_inds',
                    'clip_len',
                    'num_clips',
                    'frame_interval',
                    'fps',
                    'is_val',
                    'is_test',
                    'text',
                    'have_accident',
                ),
                meta_keys=(),
                type='PackActionInputs'),
        ],
        pipeline_video=[
            dict(
                clip_len=5,
                num_clips=30,
                test_mode=False,
                type='V6SampleFramesBeforeAccident'),
            dict(type='NativeJpegDecode'),
            dict(
                area_range=(
                    0.8,
                    1.0,
                ),
                aspect_ratio_range=(
                    1.3333333333333333,
                    1.7777777777777777,
                ),
                type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(modality='rgb', type='Flow'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                algorithm_keys=(
                    'dataset',
                    'frame_dir',
                    'filename_tmpl',
                    'img_shape',
                    'sample_idx',
                    'video_id',
                    'type',
                    'start_index',
                    'total_frames',
                    'target',
                    'abnormal_start_frame',
                    'accident_frame',
                    'frame_inds',
                    'clip_len',
                    'num_clips',
                    'frame_interval',
                    'fps',
                    'is_val',
                    'is_test',
                    'text',
                    'have_accident',
                ),
                meta_keys=(),
                type='PackActionInputs'),
        ],
        seed=42,
        split='train',
        test_mode=False,
        type='NexarSubsetDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=True, shuffle=True, type='DefaultSampler'))
train_pipeline_frame = [
    dict(
        clip_len=5,
        num_clips=30,
        test_mode=False,
        type='V6SampleFramesBeforeAccident'),
    dict(type='NativeJpegDecode'),
    dict(
        area_range=(
            0.8,
            1.0,
        ),
        aspect_ratio_range=(
            1.3333333333333333,
            1.7777777777777777,
        ),
        type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(modality='rgb', type='Flow'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(
        algorithm_keys=(
            'dataset',
            'frame_dir',
            'filename_tmpl',
            'img_shape',
            'sample_idx',
            'video_id',
            'type',
            'start_index',
            'total_frames',
            'target',
            'abnormal_start_frame',
            'accident_frame',
            'frame_inds',
            'clip_len',
            'num_clips',
            'frame_interval',
            'fps',
            'is_val',
            'is_test',
            'text',
            'have_accident',
        ),
        meta_keys=(),
        type='PackActionInputs'),
]
train_pipeline_video = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=5,
        num_clips=30,
        test_mode=False,
        type='SampleFramesBeforeAccident'),
    dict(type='DecordDecode'),
    dict(
        area_range=(
            0.8,
            1.0,
        ),
        aspect_ratio_range=(
            1.3333333333333333,
            1.7777777777777777,
        ),
        type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(modality='rgb', type='Flow'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(
        algorithm_keys=(
            'dataset',
            'frame_dir',
            'filename_tmpl',
            'img_shape',
            'sample_idx',
            'video_id',
            'type',
            'start_index',
            'total_frames',
            'target',
            'abnormal_start_frame',
            'accident_frame',
            'frame_inds',
            'clip_len',
            'num_clips',
            'frame_interval',
            'fps',
            'is_val',
            'is_test',
        ),
        meta_keys=(),
        type='PackActionInputs'),
]
val_cfg = dict(fp16=False, type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        modality='rgb',
        nexar=dict(
            ann_file='annotations.csv',
            data_root=
            '/workspace/top_v2_v6matched_fixed_3seeds/seed_42/prepared_nexar',
            filename_tmpl='{:06}.jpg',
            start_index=0),
        pipeline_frame=[
            dict(
                clip_len=5,
                num_clips=30,
                test_mode=True,
                type='V6SampleFramesBeforeAccident'),
            dict(type='NativeJpegDecode'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(modality='rgb', type='Flow'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                algorithm_keys=(
                    'dataset',
                    'frame_dir',
                    'filename_tmpl',
                    'img_shape',
                    'sample_idx',
                    'video_id',
                    'type',
                    'start_index',
                    'total_frames',
                    'target',
                    'abnormal_start_frame',
                    'accident_frame',
                    'frame_inds',
                    'clip_len',
                    'num_clips',
                    'frame_interval',
                    'fps',
                    'is_val',
                    'is_test',
                    'text',
                    'have_accident',
                ),
                meta_keys=(),
                type='PackActionInputs'),
        ],
        pipeline_video=[
            dict(
                clip_len=5,
                num_clips=30,
                test_mode=True,
                type='V6SampleFramesBeforeAccident'),
            dict(type='NativeJpegDecode'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(modality='rgb', type='Flow'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                algorithm_keys=(
                    'dataset',
                    'frame_dir',
                    'filename_tmpl',
                    'img_shape',
                    'sample_idx',
                    'video_id',
                    'type',
                    'start_index',
                    'total_frames',
                    'target',
                    'abnormal_start_frame',
                    'accident_frame',
                    'frame_inds',
                    'clip_len',
                    'num_clips',
                    'frame_interval',
                    'fps',
                    'is_val',
                    'is_test',
                    'text',
                    'have_accident',
                ),
                meta_keys=(),
                type='PackActionInputs'),
        ],
        seed=42,
        split='validation',
        test_mode=True,
        type='NexarSubsetDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    fpr_max=0.1, output_dir=None, type='SavingAnticipationMetric', vis_list=[])
val_pipeline_frame = [
    dict(
        clip_len=5,
        num_clips=30,
        test_mode=True,
        type='V6SampleFramesBeforeAccident'),
    dict(type='NativeJpegDecode'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(modality='rgb', type='Flow'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(
        algorithm_keys=(
            'dataset',
            'frame_dir',
            'filename_tmpl',
            'img_shape',
            'sample_idx',
            'video_id',
            'type',
            'start_index',
            'total_frames',
            'target',
            'abnormal_start_frame',
            'accident_frame',
            'frame_inds',
            'clip_len',
            'num_clips',
            'frame_interval',
            'fps',
            'is_val',
            'is_test',
            'text',
            'have_accident',
        ),
        meta_keys=(),
        type='PackActionInputs'),
]
val_pipeline_video = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=5,
        num_clips=30,
        test_mode=True,
        type='SampleFramesBeforeAccident'),
    dict(type='DecordDecode'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(modality='rgb', type='Flow'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(
        algorithm_keys=(
            'dataset',
            'frame_dir',
            'filename_tmpl',
            'img_shape',
            'sample_idx',
            'video_id',
            'type',
            'start_index',
            'total_frames',
            'target',
            'abnormal_start_frame',
            'accident_frame',
            'frame_inds',
            'clip_len',
            'num_clips',
            'frame_interval',
            'fps',
            'is_val',
            'is_test',
        ),
        meta_keys=(),
        type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
vis_list = []
visualizer = dict(
    type='Visualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/workspace/top_v2_v6matched_fixed_3seeds/seed_42/run'
