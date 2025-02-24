test_name = 'Test'
model = dict(type='UIEC2Net',
             get_parameter=True)
dataset_type = 'CoralDataset'

data_root_test = '/home/user/projects/ai4c/UW/DATA/test_set/'
test_ann_file_path = 'annotations_ATL_CE.csv'
images = 'images_ATL/'


img_norm_cfg = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
test_pipeline = [dict(type='LoadImageFromFile', gt_type='color', get_gt=False),
                 #dict(type='Resize', img_scale=(1024,1024), keep_ratio=True),
                 # dict(type='Pad', size_divisor=32, mode='resize'),
                 dict(type='ImageToTensor')]

usebytescale = False                                    # if use output min->0, max->255, default is False (copy from scipy=1.1.0)

data = dict(
    samples_per_gpu=4,                                  # batch size, default = 4
    workers_per_gpu=0,                                  # multi process, default = 4, debug uses 0
    val_samples_per_gpu=1,                              # validate batch size, default = 1
    val_workers_per_gpu=0,                              # validate multi process, default = 4
    test=dict(                                          # load data in test process
        type=dataset_type,
        ann_file=data_root_test + test_ann_file_path,
        img_prefix=data_root_test + images,
        pipeline=test_pipeline,
        test_mode=True))

train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SSIM', 'MSE', 'PSNR'])

loss_ssim = dict(type='SSIMLoss', window_size=11,
                 size_average=True, loss_weight=1.0)
loss_l1 = dict(type='L1Loss', loss_weight=2.0)
loss_perc = dict(type='PerceptualLoss', loss_weight=0,
                 no_vgg_instance=False, vgg_mean=False,
                 vgg_choose='conv4_3', vgg_maxpooling=False)

optimizer = dict(type='Adam', lr=1e-3, betas=[0.9, 0.999])    # optimizer with type, learning rate, and betas.

# 需要写iter
lr_config = dict(type='Epoch',          # Epoch or Iter
                 warmup='linear',       # liner, step, exp,
                 step=[100, 700],          # start with 1
                 liner_end=0.00001,
                 step_gamma=0.1,
                 exp_gamma=0.9)

# 需要写
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        dict(type='VisdomLoggerHook')
    ])

total_epoch = 1000
total_iters = None                      # epoch before iters,
work_dir = './checkpoints/UIEC2Net/1'      #
load_from = None                        # only load network parameters
resume_from = None                      # resume training
save_freq_iters = 500                   # saving frequent (saving every XX iters)
save_freq_epoch = 1                     # saving frequent (saving every XX epoch(s))
log_level = 'INFO'                      # The level of logging.

savepath = 'results/UIEC2Net'