import os.path
from glob import glob
from os.path import join

import mmcv
import pypcd
import ntpath
import numpy as np


class A92KITTI(object):
    """A9 dataset to KITTI converter.

        This class serves as the converter to change the A9 data to KITTI
        format.

        Args:
            load_dir (str): Directory to load waymo raw data.
            save_dir (str): Directory to save data in KITTI format.
            prefix (str): Prefix of filename. In general, 0 for training, 1 for
                validation and 2 for testing.
            workers (int, optional): Number of workers for the parallel process.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 prefix,
                 workers=1):
        self.selected_a9_classes = []  # TODO: Add the chosen classes
        self.a9_to_kitti_class_map = {}  # TODO: Add the mapping

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)

        # TODO: Add the parser to work through the datat sets folder

        self.label_save_dir = f'{self.save_dir}/label_'
        self.label_all_save_dir = f'{self.save_dir}/label_all'
        self.image_save_dir = f'{self.save_dir}/image_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'
        self.timestamp_save_dir = f'{self.save_dir}/timestamp'

        self.create_folder()

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        # TODO: Add list of different datasets
        mmcv.track_parallel_progress(self.convert_one, range(len(self)),
                                     self.workers)
        print('\nFinished ...')

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        pass

    def pcd_to_bin(self, file_path):
        """Convert a .pcd file to .bin file

        Args:
            file_path: Path to the .pcd file to be converted
        """
        filename = ntpath.basename(file_path)

        point_cloud = pypcd.PointCloud.from_path(file_path)
        np_x = np.array(point_cloud.pc_data['x'], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data['y'], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data['z'], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data['intensity'], dtype=np.float32)/256

        # concatenate x,y,z,intensity -> dim-4
        bin_file = np.vstack(np_x, np_y, np_z, np_i).T
        save_path = f'{self.point_cloud_save_dir}/{self.prefix}{filename}.bin'
        bin_file.tofile()

    def create_folder(self):
        """Create folder for data preprocessing."""

        dir_list1 = [
            self.calib_save_dir, self.point_cloud_save_dir,
            self.pose_save_dir, self.timestamp_save_dir
        ]
        dir_list2 = [self.image_save_dir]

        for d in dir_list1:
            mmcv.mkdir_or_exist(d)
        for d in dir_list2:
            for i in range(5):
                mmcv.mkdir_or_exist(f'{d}{str(i)}')
