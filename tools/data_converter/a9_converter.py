import json
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
                 version,
                 prefix,
                 workers=1,
                 test_mode=False):
        self.selected_a9_classes = []  # TODO: Add the chosen classes
        self.a9_to_kitti_class_map = {}  # TODO: Add the mapping

        self.is_pcd = False
        self.is_img = False
        self.is_multi_modal = False

        if version == 'point_cloud':
            self.is_pcd = True
        elif version == 'image':
            self.is_img = True
        else:
            self.is_multi_modal = True

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.version = version
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode

        self.image_save_dir = f'{self.save_dir}/image_'
        self.label_save_dir = f'{self.save_dir}/label_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.label_all_save_dir = f'{self.save_dir}/label_all'

        self.create_folder()

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        # Check if target dir exists else create
        if not os.path.exists(self.point_cloud_save_dir):
            os.makedirs(self.point_cloud_save_dir)
        dir_list = glob(os.path.join(self.load_dir, 'pcd_format'))
        mmcv.track_parallel_progress(self.convert_one, dir_list,
                                     self.workers)
        print('\nFinished ...')

    def convert_one(self, directory, directory_idx):
        """Convert action for files in directory.

        Args:
            directory (String): Directory containing the files to be converted.
        """
        # Check if directory exists else create
        target_dir = os.path.join(self.point_cloud_save_dir, directory)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        # Convert pcd to bin format
        # TODO: Add Multi Modal creation, need to analyse calib file and get the matching files
        file_list = glob(os.path.join(self.load_dir, directory))
        for file_idx, file in enumerate(file_list):
            if self.is_multi_modal:
                pass
            else:
                if self.is_pcd:
                    self.save_lidar(directory_idx, file_idx, f'{file}.pcd')

                if self.is_img:
                    self.save_image(directory_idx, file_idx, 1, f'{file}.png')

            if not self.test_mode:
                self.save_label()

    def save_image(self, directory_idx, file_idx, camera_idx, image):
        img_path = f'{self.image_save_dir}{str(camera_idx - 1)}/' + \
                   f'{self.prefix}{str(file_idx).zfill(6)}' + \
                   f'{str(directory_idx).zfill(3)}.png'
        img = mmcv.imfrombytes(image)
        mmcv.imwrite(img, img_path)

    def save_calib(self, directory_idx, file_idx, file):
        """

        """
        # TODO: Add when need multi modal data
        pass

    def save_label(self, directory_idx, file_idx, file):
        """
        Iterate through the labels for a given file and save it as kitti format
        Args:
            directory_idx:
            file_idx:
            file:

        Returns:

        """

        # Read file

        with open(file, 'r') as f:
            label_data = json.load(f)

        fp_label_all = open(
            f'{self.label_all_save_dir}/{self.prefix}' +
            f'{str(file_idx).zfill(3)}{str(directory_idx).zfill(3)}.txt', 'w+')

        # Get bounding box
        lines = []
        id_to_box = dict()
        for label in  label_data["labels"]:

            bounding_box = [
                label["box3d"]["location"]["x"] - label["box3d"]["dimension"]["length"] / 2,
                label["box3d"]["location"]["y"] - label["box3d"]["dimension"]["width"] / 2,
                label["box3d"]["location"]["x"] + label["box3d"]["dimension"]["length"] / 2,
                label["box3d"]["location"]["y"] + label["box3d"]["dimension"]["width"] / 2,
            ]
            id_to_box[label["id"]] = bounding_box

            # Not available
            truncated = 0
            occluded = 0
            alpha = -10
            length = label["box3d"]["dimension"]["length"]
            height = label["box3d"]["dimension"]["height"]
            width = label["box3d"]["dimension"]["width"]
            x = label["box3d"]["location"]["x"]
            y = label["box3d"]["location"]["y"]
            z = label["box3d"]["location"]["z"] - height / 2
            heading = label["box3d"]["orientation"]["rotationYaw"]
            line = f"{label['category']} {round(truncated, 2)} {occluded} {round(alpha, 2)} " + \
                   f"{round(bounding_box[0], 2)} {round(bounding_box[1], 2)} {round(bounding_box[2], 2)} " + \
                   f"{round(bounding_box[3], 2)} {round(height, 2)} {round(width, 2)} {round(length, 2)} " + \
                   f"{round(x, 2)} {round(y, 2)} {round(z, 2)} {round(heading, 2)}\n "

            # TODO: May consider tracking saving unique id
            # if self.save_track_id:
            #     line_all = line[:-1] + ' ' + file + ' ' + track_id + '\n'
            # else:
            #     line_all = line[:-1] + ' ' + name + '\n'
            lines.append(line)
            # TODO: Adapt to multi modal
        fp_label = open(
            f'{self.label_save_dir}{0}/{self.prefix}' +
            f'{str(file_idx).zfill(3)}{str(directory_idx).zfill(3)}.txt', 'a')
        fp_label.writelines(lines)
        fp_label.close()

        # TODO: Adapt tracking later
        # fp_label_all.write(line_all)
        #
        # fp_label_all.close()

    def save_lidar(self, directory_idx, file_idx, file):
        filename = file.split('.')[0]
        bin_format = self.pcd_to_bin(file)
        bin_format.tofile(os.path.join(self.point_cloud_save_dir, filename))

    @staticmethod
    def pcd_to_bin(file_path):
        """Convert a .pcd file to .bin file

        Args:
            file_path: Path to the .pcd file to be converted
        """

        point_cloud = pypcd.PointCloud.from_path(file_path)
        np_x = np.array(point_cloud.pc_data['x'], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data['y'], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data['z'], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data['intensity'], dtype=np.float32) / 256

        # concatenate x,y,z,intensity -> dim-4
        bin_file = np.vstack(np_x, np_y, np_z, np_i).T
        return bin_file

    def create_folder(self):
        """Create folder for data preprocessing."""

        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir, self.calib_save_dir,
                self.point_cloud_save_dir
            ]
            dir_list2 = [self.label_save_dir, self.image_save_dir]
        else:
            dir_list1 = [
                self.calib_save_dir, self.point_cloud_save_dir
            ]
            dir_list2 = [self.image_save_dir]
        for d in dir_list1:
            mmcv.mkdir_or_exist(d)
        for d in dir_list2:
            # TODO: Make this dynamic based on cameras on intersection
            if self.is_multi_modal:
                for i in range(5):
                    mmcv.mkdir_or_exist(f'{d}{str(i)}')
            else:
                pass