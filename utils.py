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
import tensorflow as tf


def clear_and_create(dir='./train_out'):
    '''
  create train_out file directory
  delete then create if already exists
  '''
    if os.path.isdir(dir):
        try:
            shutil.rmtree(dir)
            os.mkdir(dir)
        except OSError as e:
            print('Error: %s : %s' % (dir, e.strerror))
    else:
        os.mkdir(dir)


def largest_gap(indices):
    max_gap = 0
    max_gap_ind = 0
    for i in range(len(indices) - 1):
        gap = indices[i + 1] - indices[i]
        if gap > max_gap:
            max_gap = gap
            max_gap_ind = i
    return indices[max_gap_ind]


def segment_foreground(img, depth_image, is_pil=False, save=False, save_loc="", save_name="segmented"):
    ###
    # input: PIL image, PIL image of depth map
    # optional input:
    # is_pil - returns image as np if True
    # output: np array of image with black background, pil array if is_np flag set to True
    ###
    threshold = 0.2 * 1920 * 1020  # the foreground must take up at least 20% of the image
    depth_map = np.asarray(depth_image)
    hist = np.histogram(depth_map.flatten(), np.arange(np.min(depth_map), np.max(depth_map), 1))
    split = np.where(hist[0] == 0)

    if len(split[0]) != 0:
        cutoff = hist[1][split[0][0]]  # cut off as first zero value from the histogram

        foreground = depth_map < cutoff

        if cutoff > 200 or np.sum(
                foreground) < threshold:  # values determined from observation on a small subset of training data
            peaks = argrelextrema(gaussian_filter(hist[0], sigma=3),
                                  np.greater)  # smooth histogram values and extract peaks
            if len(hist[1]) < peaks[0][-1]:  # sometimes with the smoothing there is an extra value
                peak_points = hist[1][peaks[0][:-1]]
            else:
                peak_points = hist[1][peaks[0]]

            cutoff_peak = largest_gap(
                peak_points)  # determined via observation that the largest gaps between peaks generally divides foreground and background

            # Using the next minima after the cutoff peaks can result in inaccurate results for some cases (in some cases there are no clear minima after the peak)
            # The next minima is approximated by assuming the peak is symmetrical and taking the last minima before the peak and determining the half  width of the peak to estimate the full width of the peak

            minima = argrelextrema(hist[0][:np.where(hist[1] == cutoff_peak)[0][0]], np.less)[0]
            if len(minima) != 0:  # no else statment in the case that there is not multiple peaks in the overall but the foreground is smaller than the threshold
                half_bump_width = cutoff_peak - hist[1][minima[-1]]

                cutoff = cutoff_peak + half_bump_width

                foreground = depth_map < cutoff

        foreground_img = img * np.stack([foreground, foreground, foreground], axis=2)
        if save:
            foreground_img_pil = Image.fromarray(foreground_img)
            foreground_img_pil.save('out' + save_name + ".png")

        if is_pil:
            return Image.fromarray(foreground_img)
        else:
            return foreground_img

    else:
        if save:
            img.save('out' + save_name + ".png")
        if is_pil:
            return img
        else:
            return np.asarray(img)


def get_rect(image=None, mask=None, draw=True):
    '''
    draw returns an image where a red rectangle is fit around the mask
    when false returns the cropped rectangular image...
    '''
    index_true = np.argwhere(mask == 1)
    start_point = (np.min(index_true[:, 1]), np.min(index_true[:, 0]))
    end_point = (np.max(index_true[:, 1]), np.max(index_true[:, 0]))

    if draw:
        # Red color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        return cv2.rectangle(image, start_point, end_point, color, thickness)
    else:
        # crop the image per the minimum rectangle that fits to the mask
        # area in the rectangle that is not in the superpixel is black...
        # x,y,w,h = cv2.boundingRect(mask)
        # roi = image[y:y+h, x:x+w, :]
        roi = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        return roi

def decision(probability):
    '''
    return true with probability of probability
    :param probability:
    :return:
    '''
    return random.random() < probability

def whatComponent(label_mask, mask, cluster, threshold=0.75):
    """
    This function takes the mask of the superpixel and the label...
    Calculate how many pixels in the mask belong to each class as a ratio to the total number of pixels
    If one ratio is above threshold return the component index
    index - component name - colour
    0 - wall - (202,150,150)
    1 - beam - (198, 186, 100)
    2 - column - (167, 183, 186)
    3 - window frame - (255, 255, 133)
    4 - window pane - (192, 192, 206)
	5 - balcony - (32, 80, 160)
	6 - slab - (193, 134, 1)
	100 - ignore - (70,70,70)
    :param label_mask:
    :param mask:
    :param cluster:
    :param threshold:
    :return: returns a index above... If all values below threshold then return code 200. If assert error return 300
    """
    reduced_label_mask = label_mask[:, :, 0]*mask

    # Get the total area of the cluster
    area_cluster = np.sum(mask)

    num_wall = np.count_nonzero(reduced_label_mask == 150)
    num_beam = np.count_nonzero(reduced_label_mask == 100)
    num_column = np.count_nonzero(reduced_label_mask == 186)
    num_window_frame = np.count_nonzero(reduced_label_mask == 133)
    num_window_pane = np.count_nonzero(reduced_label_mask == 206)
    num_balcony = np.count_nonzero(reduced_label_mask == 160)
    num_slab = np.count_nonzero(reduced_label_mask == 1)
    num_ignore = np.count_nonzero(reduced_label_mask == 70)

    try:
        assert np.sum((num_balcony, num_slab, num_ignore, num_beam, num_wall, num_column, num_window_pane, num_window_frame)) == area_cluster
    except AssertionError:
        print("Something is not adding up.")
        return 300

    threshold_pixs = int(threshold*area_cluster)
    if num_wall > threshold_pixs:
        if decision(1):
            return 0
        else:
            return 200
    elif num_beam > threshold_pixs:
        return 1
    elif num_column > threshold_pixs:
        if decision(1):
            return 2
        else:
            return 200
    elif num_window_frame > threshold_pixs:
        if decision(1):
            return 3
        else:
            return 200
    elif num_window_pane > threshold_pixs:
        if decision(1):
            return 4
        else:
            return 200
    elif num_balcony > threshold_pixs:
        return 5
    elif num_slab > threshold_pixs:
        return 6
    elif num_ignore > threshold_pixs:
        #if np.average(cluster) <= 1:
        #    return 200
        #else:
        return 100
    else:
        return 200

def preprocessImages (img, imgsize = 224):
    '''
    Run raw images through here to be processed then send into model.
    This function normalizes img in ranges [-1, 1] and resizes to them to imgsize
    :param img:
    :param imgsize: 224 (default) depends on trained model
    :return: img normalized between -1 and 1 and resized to imgsize x imgsize
    '''
    # Assume in float32 format
    img = (img / 127.5) - 1
    img = cv2.resize(img, (imgsize, imgsize))
    img = img.reshape(1, imgsize, imgsize, 3)
    return img

### let's take the save_mask from USP repo