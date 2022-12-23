# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm


PALETTE = ([ 148, 218, 255 ],  # light blue
        [  85,  85,  85 ],  # almost black
        [ 200, 219, 190 ],  # light green
        [ 166, 133, 226 ],  # purple    
        [ 255, 171, 225 ],  # pink
        [  40, 150, 114 ],  # green
        [ 234, 144, 133 ],  # orange
        [  89,  82,  96 ],  # dark gray
        [ 255, 255,   0 ],  # yellow
        [ 110,  87, 121 ],  # dark purple
        [ 205, 201, 195 ],  # light gray
        [ 212,  80, 121 ],  # medium red
        [ 159, 135, 114 ],  # light brown
        [ 102,  90,  72 ],  # dark brown
        [ 255, 255, 102 ],  # bright yellow
        [ 251, 247, 240 ])  # almost white


def label_to_color(gt_img):
    """
    Given a ground-truth image, return an rgb image which is easier to interpret visually
    """
    width = gt_img.shape[0]
    height = gt_img.shape[1]
    bgr = np.zeros((width, height, 3), dtype=np.uint8)
    for k, color in enumerate(PALETTE):
        bgr[gt_img == k] = [color[2], color[1], color[0]]
    # set background black
    bgr[gt_img == 255] = [0, 0, 0]
    return bgr


def plot_semseg_comparison(image, target, prediction, output_filename):    
    fig, ax = plt.subplots(1, 3, figsize=(50, 50))
    ax[0].imshow(image)
    ax[0].set_title("Raw image + Prediction")
    ax[1].imshow(prediction)
    ax[1].set_title("Prediction")
    ax[2].imshow(target)
    ax[2].set_title("Ground-truth")
    fig.tight_layout()
    fig.savefig(output_filename)
    plt.close()
    

def load_image(
        filename,
        tonemap_fct_16_bit = lambda x: cv2.sqrt(x.astype(np.float32)).astype(np.uint8)):
    """
    This function reads a image from the given filename and converts it to
    8-bit if it is a 16-bit image.
    """
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image.dtype == np.uint16:
        return tonemap_fct_16_bit(image)
    return image


def blend_gt(img_gray, img_labels, alpha=0.5):
    """
    Given an input image and a color ground-truth or prediction image, blend them using the alpha channel
    """
    img_arr_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    composite = cv2.addWeighted(img_arr_color, 1.0, img_labels, alpha, 1.0)
    return composite


def main(inputs_path, annotations_path, output_dir, gt_path=None, debug=False):
    if debug and gt_path is None:
        print(f"--debug needs a ground-truth path (--gt_path)")
        sys.exit()
    os.makedirs(output_dir, exist_ok=True)
    annots = os.listdir(annotations_path)
    for fname in tqdm(annots):
        annot_path = os.path.join(annotations_path, fname)
        input_path = os.path.join(inputs_path, fname)
        output_path = os.path.join(output_dir, fname)
        labeled_img = load_image(annot_path)
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_RGB2BGR)
        input_img = load_image(input_path)
        comp = blend_gt(input_img, labeled_img)
        cv2.imwrite(output_path, comp)
        if debug:
            gt_img = label_to_color(load_image(os.path.join(gt_path, fname)))
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            output_filename = output_path.replace(".png", "_debug.png")
            plot_semseg_comparison(comp, gt_img, labeled_img, output_filename)


if __name__ == "__main__":
    fire.Fire(main)
