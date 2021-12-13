from core.Datasets.base_dataset import BaseDataset
from core.Datasets.builder import DATASETS
import copy
import numpy as np
import pandas as pd
import os
import torchvision
from PIL import Image

@DATASETS.register_module()
class CoralDataset(BaseDataset):
    def __init__(self, **kwargs):
        super(CoralDataset, self).__init__(**kwargs)
        self.data_infos = self.load_annotations()
        self.encoding = {'Algae':0,'Hard Coral':1,'Other':2,'Soft Coral':3,'Other Invertebrates':4,'Sponge':5}
        self.classes = self.encoding.keys()
        self._set_group_flag()

    def load_annotations(self):
        data_infos = []
        self.annotations = pd.read_csv(self.ann_file)
        #self.cleaned_annotations = self.annotations
        #self.cleaned_annotations["original_index"] = self.annotations.index
        #indeces_to_drop = []
        #for index, row in self.annotations.iterrows():
        #    if not os.path.exists(os.path.join(self.img_prefix, str(self.annotations.iloc[index, 0]) +".jpg")):
        #        indeces_to_drop.append(index)
        #self.cleaned_annotations = self.cleaned_annotations.drop(self.cleaned_annotations.index[indeces_to_drop]).reset_index()
        for index, annotation in self.annotations.iterrows():
                try:
                    data_infos.append({
                        "image_path": os.path.join(self.img_prefix, str(self.annotations.iloc[index, 0]) +".jpg"),
                        "crop_left": int(self.annotations.iloc[index, 1]) - 112,
                        "crop_top": int(self.annotations.iloc[index, 2]) - 112,
                        "image_id": str(self.annotations.iloc[index, 0]) + "_" + str(index) + "_" + str(self.annotations.iloc[index, 4]).replace("/", ""),
                        "label": str(self.annotations.iloc[index, 4]),
                        "ann_id": index
                    })
                except ValueError as e:
                    print(e)
                    continue
        return data_infos

    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of the training batch data.

        Returns:
            dict: Returned training batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index for getting each testing batch.

        Returns:
            Tensor: Returned testing batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        img = Image.open(results["image_path"]).convert("RGB")
        img = torchvision.transforms.functional.crop(img,results["crop_top"],results["crop_left"],224,224)
        img = torchvision.transforms.ToTensor()(img).cuda()
        results["image"] = img

        return results

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            # if img_info['width'] / img_info['height'] > 1:
            self.flag[i] = 1