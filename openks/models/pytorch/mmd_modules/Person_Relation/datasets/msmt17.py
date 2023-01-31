
import glob
import re
import time
import random
import os.path as osp
import numpy as np

from .bases import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'MSMT17_V1'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17, self).__init__()
        self.with_time = True
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        if self.with_time == True:
            self.list_train_path = osp.join(self.dataset_dir, 'list_train_withtime.txt')
            #self.list_train_locus = osp.join(self.dataset_dir, 'train_locus.txt')
            self.list_val_path = osp.join(self.dataset_dir, 'list_val_withtime.txt')
            self.list_query_path = osp.join(self.dataset_dir, 'list_query_withtime.txt')
            self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery_withtime.txt')
        else:
            self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
            self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
            self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
            self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        if self.with_time == True:
            train = self._process_dir_with_time(self.train_dir, self.list_train_path)
            val = self._process_dir_with_time(self.train_dir, self.list_val_path)
            train += val
            query = self._process_dir_with_time(self.test_dir, self.list_query_path)
            gallery = self._process_dir_with_time(self.test_dir, self.list_gallery_path)
        else:
            train = self._process_dir(self.train_dir, self.list_train_path)
            val = self._process_dir(self.train_dir, self.list_val_path)
            train += val
            query = self._process_dir(self.test_dir, self.list_query_path)
            gallery = self._process_dir(self.test_dir, self.list_gallery_path)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, self.pid_begin +pid, camid-1, 1))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        #for idx, pid in enumerate(pid_container):
        #    assert idx == pid, "See code comment for explanation"
        return dataset

    """
    def _process_dir_with_time(self, dir_path, list_path):
        basetime = '2022-01-01:00:00:00'
        basetimeArray = time.strptime(basetime, "%Y-%m-%d:%H:%M:%S")
        basetimeStamp = int(time.mktime(basetimeArray))

        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, times, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            #timesArray = time.strptime(times, "%Y-%m-%d:%H:%M:%S")
            #timesStamp = int(time.mktime(timesArray))
            #newtimesStamp = timesStamp - basetimeStamp
            newtimesStamp = times
            dataset.append((img_path, self.pid_begin +pid, camid-1, 1, newtimesStamp))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        #for idx, pid in enumerate(pid_container):
        #    assert idx == pid, "See code comment for explanation"
        return dataset
    """

    def _process_dir_with_time(self, dir_path, list_path, locus_path=None):
        basetime = '2022-01-01:00:00:00'
        basetimeArray = time.strptime(basetime, "%Y-%m-%d:%H:%M:%S")
        basetimeStamp = int(time.mktime(basetimeArray))
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        if locus_path:
            with open(locus_path, 'r') as locus_txt:
                locus_lines = locus_txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        num = 0
        MAX_LEN = 20
        for img_idx, img_info in enumerate(lines):
            img_path, times, x, y, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            x = float(x)
            y = float(y)
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            timesArray = time.strptime(times, "%Y-%m-%d:%H:%M:%S")
            times = int(time.mktime(timesArray))
            newtimesStamp = (times - basetimeStamp) / 60
            #newtimesStamp = times / 60

            dataset.append((img_path, self.pid_begin + pid, camid - 1, 1, newtimesStamp, x, y))
            pid_container.add(pid)
            cam_container.add(camid)
            num += 1
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        # for idx, pid in enumerate(pid_container):
        #    assert idx == pid, "See code comment for explanation"
        return dataset
