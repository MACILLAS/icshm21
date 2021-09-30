import tensorflow as tf
import cv2
from utils import clear_and_create, segment_foreground, get_rect, preprocessImages
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

def saveMask (row, col, chn, im_target, name): # Code borrowed from USP
    # Define label_colours
    label_colours = np.random.randint(255, size=(100, 3))
    im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(row, col, chn).astype(np.uint8)
    # Save mask (colourized)
    cv2.imwrite("output" + str(name) + ".png", im_target_rgb)

if __name__ == "__main__":
    # VARIABLE DECLARATION
    idx = 0

    conf_threshold = 0.7

    # load in the list of test set
    test_list = pd.read_csv('test.csv', header=None)
    img_file = test_list[0].iloc[idx]  # image at index 0 of the test set

    model = tf.keras.models.load_model('./saved_model/trained_model.h5')

    # read RGB image
    img = cv2.imread(os.path.join(IMGDIR, img_file), -1)
    # read depth image
    depth_img = cv2.imread(os.path.join(LABELDIR, DEPTHDIR, img_file), -1)

    # get the foreground img
    #try:
    #    foregnd_img = segment_foreground(img, depth_img, is_pil=False, save=False)
    #except Exception:
    #    print("ERROR")
    #    raise ChildProcessError
    foregnd_img = img

    # generate superpixels
    try:
        # generate superpixels (SEEDS algorithm)
        seeds = cv2.ximgproc.createSuperpixelSEEDS(foregnd_img.shape[1], foregnd_img.shape[0], foregnd_img.shape[2], 300, 5, 3, 15, True)
        seeds.iterate(foregnd_img, 50)  # The input image size must be the same as the initial shape, the number of iterations is 10
    except Exception:
        print("SuperpixelSEEDS Error")
        raise ChildProcessError

    mask_seeds = seeds.getLabelContourMask()
    label_seeds = seeds.getLabels()
    number_seeds = seeds.getNumberOfSuperpixels()
    mask_inv_seeds = cv2.bitwise_not(mask_seeds)
    img_seeds = cv2.bitwise_and(foregnd_img, foregnd_img, mask=mask_inv_seeds)

    # We use variable to start building up the full mask
    full_segmentation = np.zeros(foregnd_img.shape[:2])

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

        #cropped_file = str(v) + '_' + img_file
        # print(v)
        # print(cluster.shape)
        # print(cropped_file)

        # preprocess cropped for inference
        # get the predicted class for that cluster
        cropped_pre = preprocessImages(cropped)
        # if np.mean(cropped) < 0.5 : pred_class = 9, pred_class_conf = 1
        if np.mean(cropped) < 1:
            pred_class = 9
            pred_class_conf = 1
        else:
            pred_class, pred_class_conf = predict_component(model, cropped_pre)

        ### DEBUG CODE ###
        #print(f"Predicted class: {pred_class} with Confidence of: {pred_class_conf}")

        # when the predicted class confidence is greater than the confidence threshold...
        # 'add' the semantic segmentation to the full_segmentation
        if pred_class_conf >= conf_threshold:
            full_segmentation = full_segmentation + mask*pred_class

    ### DEBUG CODE ###
    saveMask(foregnd_img.shape[0], foregnd_img.shape[1], 3, full_segmentation.astype(int), 'debug_segmentation')