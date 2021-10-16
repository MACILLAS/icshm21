import numpy
import tensorflow as tf
import cv2
from utils import clear_and_create, segment_foreground, get_rect, preprocess_images
import numpy as np
import pandas as pd
import os

### Directory Declaration
IMGDIR = "./image"
LABELDIR = "./label"
DEPTHDIR = "./depth"


def predict_component(model=None, img=None):
    """
    returns the class index and confidencce of the prediction
    :param model:
    :param img:
    :return:
    """
    prediction = model.predict(img)
    pred_class = np.argmax(prediction)
    pred_class_conf = np.max(prediction)
    return pred_class, pred_class_conf


def save_mask(row: int, col: int, chn: int, im_target, save_dir: str):  # Code borrowed from USP
    # Define label_colours
    label_colours = np.random.randint(255, size=(100, 3))
    im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(row, col, chn).astype(np.uint8)
    # Save colourized mask
    cv2.imwrite("debug_"+save_dir, im_target_rgb)


def segment_image(conf_threshold=0.8, img=None, model=None, seeds=None):
    """

    :param conf_threshold:
    :param img:
    :param model:
    :param seeds:
    :return:
    """
    # We use variable to start building up the full mask
    full_segmentation = np.zeros(img.shape[:2])
    # Take an image split it into superpixels then run classifier on each superpixel
    mask_seeds = seeds.getLabelContourMask()
    label_seeds = seeds.getLabels()
    number_seeds = seeds.getNumberOfSuperpixels()
    mask_inv_seeds = cv2.bitwise_not(mask_seeds)
    img_seeds = cv2.bitwise_and(img, img, mask=mask_inv_seeds)

    # loop through each superpixel
    for v in np.unique(label_seeds):
        # construct a mask for the segment so we can compute image statistics for *only* the masked region
        mask = np.zeros(img.shape[:2])
        # mask is the same size of the image
        # where the superpixel is the mask value '1'
        mask[label_seeds == v] = 1
        cluster = (img * np.stack([mask, mask, mask], axis=2))
        # get cropped small image
        cropped = get_rect(cluster, mask, False)

        # cropped_file = str(v) + '_' + img_file
        # print(v)
        # print(cluster.shape)
        # print(cropped_file)

        # preprocess cropped for inference
        # get the predicted class for that cluster
        cropped_pre = preprocess_images(cropped, imgsize=71)
        # if np.mean(cropped) < 0.5 : pred_class = 9, pred_class_conf = 1

        # if np.mean(cropped) < 1:
        #     pred_class = 9
        #     pred_class_conf = 1
        # else:
        #    pred_class, pred_class_conf = predict_component(model, cropped_pre)
        pred_class, pred_class_conf = predict_component(model, cropped_pre)

        ### DEBUG CODE ###
        # print(f"Predicted class: {pred_class} with Confidence of: {pred_class_conf}")

        # when the predicted class confidence is greater than the confidence threshold...
        # 'add' the semantic segmentation to the full_segmentation
        if pred_class_conf >= conf_threshold:
            full_segmentation = full_segmentation + mask * pred_class
        else:
            full_segmentation = full_segmentation + mask * 200

    # Current Class Mapping  --> Submission Mapping
    # 0 balcony --> 5
    # 1 is beam --> 1
    # 2 is ignore --> 100
    # 3 is wall --> 0
    # 4 is windowframe --> 3
    # 5 is windowpane --> 4
    # 200 is uncertain --> 200
    balcony_mask = (full_segmentation == 0) * 5
    beam_mask = (full_segmentation == 1) * 1
    ignore_mask = (full_segmentation == 2) * 100
    wall_mask = (full_segmentation == 3) * 0
    window_frame_mask = (full_segmentation == 4) * 3
    window_pane_mask = (full_segmentation == 5) * 4
    uncertain_mask = (full_segmentation == 200) * 99

    # Recombine Masks
    post_processed_seg = balcony_mask + beam_mask + ignore_mask + wall_mask + window_frame_mask + window_pane_mask + uncertain_mask
    return post_processed_seg


def main(conf_threshold=0.8, files='test.csv', model_dir='./saved_model/trained_model.h5'):
    # create empty directory called ./kaggle_label
    clear_and_create('./kaggle_label')
    # create subdirectory ./kaggle_label/component
    clear_and_create('./kaggle_label/component')
    clear_and_create('./debug_output')
    # set confidence threshold
    conf_threshold = conf_threshold
    # read in files in csv with pandas as test_list
    test_list = pd.read_csv(files, header=None)
    # load the model from model directory (expecting in h5 format)
    model = tf.keras.models.load_model(model_dir)

    # loop through every image in test set
    for idx in range(test_list.size):
        print(idx)
        img_file = test_list[0].iloc[idx]  # image at index idx of set
        # read RGB image in image file to img
        img = cv2.imread(os.path.join(IMGDIR, img_file), -1)

        # generate superpixels
        try:
            # generate superpixels (SLIC algorithm)
            seeds = cv2.ximgproc.createSuperpixelSLIC(img, region_size=50)
            seeds.iterate(50)  # The input image size must be the same as the initial shape, the number of iterations is 10
        except Exception:
            print("SuperpixelSLIC Error")
            #raise ChildProcessError

        post_processed_seg = segment_image(conf_threshold=conf_threshold, img=img, model=model, seeds=seeds)
        ## DEBUG ##
        save_mask(img.shape[0], img.shape[1], 3, post_processed_seg.astype(int), os.path.join('debug_output', img_file))
        # save the image in the kaggle_label directory
        cv2.imwrite(os.path.join('./kaggle_label/component', img_file), post_processed_seg)


if __name__ == "__main__":
    main()
