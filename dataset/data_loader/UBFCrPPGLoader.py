"""The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
from torchvision import transforms
import random
class UBFCrPPGLoader(BaseLoader):
    """The data loader for the UBFC-rPPG dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an UBFC-rPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- subject1/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |   |-- subject2/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |...
                     |   |-- subjectn/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def calculate_brightness(frame):
        """Compute the brightness of a given frame."""
        return np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    def augment_brightness(frame, brightness_factor):
        return np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)


    def generate_constant_transform(self):
        pure_brightness_min, pure_brightness_max = 25, 100
        smartphone_brightness_min, smartphone_brightness_max = 90, 180
        brightness_lower_factor = smartphone_brightness_min / pure_brightness_max
        brightness_upper_factor = smartphone_brightness_max / pure_brightness_min
        brightness_const = random.uniform(brightness_lower_factor, brightness_upper_factor)

        pure_saturation_min, pure_saturation_max = 20, 145
        smartphone_saturation_min, smartphone_saturation_max = 82, 135
        saturation_lower_factor = smartphone_saturation_min / pure_saturation_max
        saturation_upper_factor = smartphone_saturation_max / pure_saturation_min
        saturation_const = random.uniform(saturation_lower_factor, saturation_upper_factor)

        constant_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(
                brightness=brightness_const,
                saturation=saturation_const,
                hue=(-0.1,0.1)
            ),
            transforms.ToTensor()
          ])
        return constant_transform

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        ubfc_brightness_min, ubfc_brightness_max = 115, 145
        smartphone_brightness_min, smartphone_brightness_max = 90, 180
        brightness_lower_factor = smartphone_brightness_min / ubfc_brightness_max
        brightness_upper_factor = smartphone_brightness_max / ubfc_brightness_min
        brightness_const = random.uniform(brightness_lower_factor, brightness_upper_factor)


        ubfc_saturation_min, ubfc_saturation_max = 95, 185
        smartphone_saturation_min, smartphone_saturation_max = 82, 135
        saturation_lower_factor = smartphone_saturation_min / ubfc_saturation_max
        saturation_upper_factor = smartphone_saturation_max / ubfc_saturation_min
        saturation_const = random.uniform(saturation_lower_factor, saturation_upper_factor)

        constant_transform = self.generate_constant_transform()

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(
                os.path.join(data_dirs[i]['path'],"vid.avi"))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(
                os.path.join(data_dirs[i]['path'],"ground_truth.txt"))
        
        augmented_frames = self.read_video(
                os.path.join(data_dirs[i]['path'],"vid.avi"), transform=constant_transform)
        

        print("Original shape:", frames.shape)
        print("Transformed shape:", augmented_frames.shape)
        

        augmented_frames = augmented_frames.astype('uint8')

        # Check if dimensions match before concatenation
        if frames.shape[1:] != augmented_frames.shape[1:]:
            raise ValueError("Dimensions of original and augmented frames must match.")
        if frames.dtype != augmented_frames.dtype:
            raise ValueError("Data types of original and augmented frames must match.", frames.dtype)
        
        

        # Combine original and augmented frames
        combined_frames = np.concatenate((frames, augmented_frames), axis=0)
    
        # Duplicate labels to match the combined frames
        duplicated_bvps = np.concatenate((bvps, bvps), axis=0)
    
        # Ensure the label length matches the combined frames length
        target_length = combined_frames.shape[0]
        duplicated_bvps = BaseLoader.resample_ppg(duplicated_bvps, target_length)
    
        frames_clips, bvps_clips = self.preprocess(
            combined_frames, duplicated_bvps, config_preprocess)
    
        input_name_list, label_name_list = self.save_multi_process(
            frames_clips, bvps_clips, saved_filename)
    
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file, transform=None):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            if transform is not None:
                frame = transform(frame)
                frame = frame.permute(1,2,0).numpy()
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)
