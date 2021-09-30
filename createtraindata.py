import numpy as np
import pandas as pd
import cv2
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter
import random
import time

from utils import clear_and_create, segment_foreground, get_rect, whatComponent

IMGDIR = "./image"
LABELDIR = "./label"
DEPTHDIR = "./depth"
TRAINDIR = './train'
COMPDIR = './component'
WALLDIR = os.path.join(TRAINDIR, './wall')
BEAMDIR = os.path.join(TRAINDIR, "./beam")
COLDIR = os.path.join(TRAINDIR, "./column")
WINFRAMEDIR = os.path.join(TRAINDIR, "./windowframe")
WINPANEDIR = os.path.join(TRAINDIR, "./windowpane")
BALCDIR = os.path.join(TRAINDIR, "./balcony")
SLABDIR = os.path.join(TRAINDIR, "./slab")
IGNOREDIR = os.path.join(TRAINDIR, "./ignore")

if __name__ == "__main__":
    train_list = pd.read_csv('train.csv', header=None)
    test_list = pd.read_csv('test.csv', header=None)

    clear_and_create(TRAINDIR)
    clear_and_create(WALLDIR)
    clear_and_create(BEAMDIR)
    #clear_and_create(COLDIR)
    clear_and_create(WINFRAMEDIR)
    clear_and_create(WINPANEDIR)
    clear_and_create(BALCDIR)
    #clear_and_create(SLABDIR)
    clear_and_create(IGNOREDIR)

    # Number of Entries
    num_train = len(train_list.index)

    # Loop through each index in train_list
    for idx in range(0, 3805):
        print(idx)
        # img_file
        img_file = train_list[0].iloc[idx]
        # read RGB image
        img = cv2.imread(os.path.join(IMGDIR, img_file), -1)
        #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        # read depth image
        #depth_img = cv2.imread(os.path.join(LABELDIR, DEPTHDIR, img_file), -1)
        #try:
        #    # foreground img
        #    foregnd_img = segment_foreground(img, depth_img, is_pil=False, save=False)
        #except Exception:
        #    print("ERROR")
        #    continue
        foregnd_img = img
        # component_label
        component_label = cv2.imread(os.path.join(LABELDIR, COMPDIR, img_file))
        try:
            # generate superpixels (SEEDS algorithm)
            #seeds = cv2.ximgproc.createSuperpixelSEEDS(foregnd_img.shape[1], foregnd_img.shape[0], foregnd_img.shape[2], 300, 5, 3, 15, True)
            seeds = cv2.ximgproc.createSuperpixelSLIC(foregnd_img, region_size=50)
            seeds.iterate(50)  # The input image size must be the same as the initial shape, the number of iterations is 10
        except Exception:
            print("SuperpixelSEEDS Error")
            continue
        mask_seeds = seeds.getLabelContourMask()
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()
        mask_inv_seeds = cv2.bitwise_not(mask_seeds)
        img_seeds = cv2.bitwise_and(foregnd_img, foregnd_img, mask=mask_inv_seeds)

        # loop through each superpixel
        for v in np.unique(label_seeds):
            # construct a mask for the segment so we can compute image statistics for *only* the masked region
            mask = np.zeros(foregnd_img.shape[:2])
            # mask is the same size of the image
            # where the superpixel is the mask value '1'
            mask[label_seeds == v] = 1
            cluster = (foregnd_img * np.stack([mask, mask, mask], axis=2))
            # get cropped small image
            cropped = get_rect(cluster, mask, False)
            cropped_file = str(v) + '_' + img_file
            # print(v)
            # print(cluster.shape)
            # print(cropped_file)

            comp_indx = whatComponent(label_mask=component_label, mask=mask, cluster=cluster, threshold=0.85)
            '''
            0 - wall - (202, 150, 150)
            1 - beam - (198, 186, 100)
            2 - column - (167, 183, 186)
            3 - window frame - (255, 255, 133)
            4 - window pane - (192, 192, 206)
            5 - balcony - (32, 80, 160)
            6 - slab - (193, 134, 1)
            100 - ignore - (70, 70, 70)
            200 - unsure
            300 - failure
            '''
            if comp_indx == 0:
                outfile = os.path.join(WALLDIR, cropped_file)
            elif comp_indx == 1 or comp_indx == 2:
                outfile = os.path.join(BEAMDIR, cropped_file)
            #elif comp_indx == 2:
            #    outfile = os.path.join(COLDIR, cropped_file)
            elif comp_indx == 3:
                outfile = os.path.join(WINFRAMEDIR, cropped_file)
            elif comp_indx == 4:
                outfile = os.path.join(WINPANEDIR, cropped_file)
            elif comp_indx == 5:
                outfile = os.path.join(BALCDIR, cropped_file)
            elif comp_indx == 6:
                #outfile = os.path.join(SLABDIR, cropped_file)
                continue
            elif comp_indx == 100:
                outfile = os.path.join(IGNOREDIR, cropped_file)
            else:
                continue

            try:
                cv2.imwrite(outfile, cropped)
            except Exception:
                print("ERROR: " + cropped_file)

    '''

    # loop through each index in train_list
    for idx in range(0, 100):  # replace 1 with num_train
        start = time.time()
        print(idx)
        # get filename of the image
        img_file = train_list[0].iloc[idx]
        # read RGB image
        img = cv2.imread(os.path.join(img_dir, img_file))
        print(img.shape)
        # read depth image
        depth_img = cv2.imread(os.path.join(label_dir, depth_dir, img_file), -1)  # load as-is
        # foreground img
        foregnd_img = segment_foreground(img, depth_img, is_pil=False, save=False)

        # generate superpixels (SEEDS algorithm)
        seeds = cv2.ximgproc.createSuperpixelSEEDS(foregnd_img.shape[1], foregnd_img.shape[0], foregnd_img.shape[2], 200, 5, 3, 15, True)
        seeds.iterate(foregnd_img, 25)  # The input image size must be the same as the initial shape, the number of iterations is 10

        # generate superpixels (SLIC algorithm)
        # seeds = cv2.ximgproc.createSuperpixelSLIC(foregnd_img, region_size=50, ruler=20)
        # seeds.iterate(25)

        mask_seeds = seeds.getLabelContourMask()
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()
        mask_inv_seeds = cv2.bitwise_not(mask_seeds)
        img_seeds = cv2.bitwise_and(foregnd_img, foregnd_img, mask=mask_inv_seeds)

        # loop through each superpixel
        for v in np.unique(label_seeds):  # replace with np.unique(label_seeds)
            # construct a mask for the segment so we can compute image statistics for *only* the masked region
            mask = np.zeros(foregnd_img.shape[:2])
            mask[label_seeds == v] = 1
            cluster = (foregnd_img * np.stack([mask, mask, mask], axis=2))
            # get cropped small image
            cropped = get_rect(cluster, mask, False)
            cropped_file = str(v) + '_' + img_file
            # print(v)
            # print(cluster.shape)
            # print(cropped_file)
            crack_bool = is_crack(idx, mask)
            rebar_bool = is_rebar(idx, mask)
            spall_bool = is_spall(idx, mask)

            if not crack_bool and not rebar_bool and not spall_bool:
                rand_num = random.randrange(0, 20, 1)
                if rand_num < 1:
                    # entry = pd.DataFrame({'file':cropped_file, 'crack':crack_bool, 'spall':spall_bool, 'rebar':rebar_bool, 'other':[True]})
                    # label_df = label_df.append(entry, ignore_index=True)
                    try:
                        cv2.imwrite(os.path.join(train_other, cropped_file), cropped)
                    except Exception:
                        print("ERROR: " + cropped_file)
            elif crack_bool:
                try:
                    cv2.imwrite(os.path.join(train_crack, cropped_file), cropped)
                except Exception:
                    print("ERROR: " + cropped_file)
            elif rebar_bool:
                try:
                    cv2.imwrite(os.path.join(train_rebar, cropped_file), cropped)
                except Exception:
                    print("ERROR: " + cropped_file)
            else:
                rand_num = random.randrange(0, 10, 1)
                if rand_num < 5:
                    try:
                        cv2.imwrite(os.path.join(train_spall, cropped_file), cropped)
                    except Exception:
                        print("ERROR: " + cropped_file)

    end = time.time()
    ex_time = str(end - start)

    print("Execution Time(s): " + ex_time)
    label_df.to_csv('train_class.csv', index=False)
    '''

