import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import torch
import time
import os.path as osp
from torch.nn.parallel import DataParallel
from utils import Config
from core.Models import build_network
from core.Datasets import build_dataset, build_dataloader
from core.Optimizer import build_optimizer, build_scheduler
from utils import (mkdir_or_exist, get_root_logger,
                      save_epoch, save_latest, save_item, normimage_test,
                      resume, load, normPRED)

from utils.save_image import (save_image, normimage,
                              save_ensemble_image, save_ensemble_image_8)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',type=str,
                        default='config/CoralUIEC2Net.py',
                        help='train config file path')
    parser.add_argument('--load_from',
                        default='checkpoints/latest.pth',
                        help='the dir to save logs and models,')
    parser.add_argument('--savepath', help='the dir to save logs and models,')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        default=1,
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args

class ModelLoader:
    def __init__(self) -> None:
        args = parse_args()
        self.cfg = Config.fromfile(args.config)
        self.model = build_network(self.cfg .model, train_cfg=self.cfg .train_cfg, test_cfg=self.cfg .test_cfg)
        load(self.cfg .load_from, self.model, None)
        if torch.cuda.is_available():
            # model = DataParallel(model.cuda(), device_ids=cfg.gpu_ids)
            self.model = self.model.cuda()
        self.model.eval()
    
    def prediction(self, image):
        save_path = 'temp'
        mkdir_or_exist(save_path)
        save_cfg = False
        for i in range(len(self.cfg .test_pipeline)):
            if 'Normalize' == self.cfg .test_pipeline[i].type:
                save_cfg = True
        with torch.no_grad():
            out_rgb = self.model(image)
        rgb_numpy = normimage_test(out_rgb, save_cfg=save_cfg, usebytescale=self.cfg .usebytescale)
        outsavepath = osp.join(save_path,  'result.png')
        # inputsavepath = osp.join(save_path, 'result_input.png')

        # save_image(input_numpy, inputsavepath)
        save_image(rgb_numpy, outsavepath, usebytescale=self.cfg.usebytescale)
        return rgb_numpy
