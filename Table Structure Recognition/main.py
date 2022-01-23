#!/usr/bin/env python
import glob
import os

import cv2
import lxml.etree as etree
from mmdet.apis import inference_detector, init_detector
from pdf2image import convert_from_path

from Functions.blessFunc import borderless
from border import border

SCRIPTS_LOCATION: str = "/home/makn/workspace-uni/CascadeTabNetTests"
CASCADE_TAB_NET_REPO_LOCATION: str = SCRIPTS_LOCATION + "/CascadeTabNet"
VISUALISATION_LOCATION: str = "/home/makn/Downloads/sample-tables/ba_test_tables"

# To Do
image_path = '/home/makn/Downloads/sample-tables/invoice.jpg'
xmlPath = '/home/makn/Downloads/sample-xml/'

config_fname = CASCADE_TAB_NET_REPO_LOCATION + "/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py"
checkpoint_file = SCRIPTS_LOCATION + "/epoch_36.pth"

model = init_detector(config_fname, checkpoint_file)


def process_image(image_here):
    result = inference_detector(model, image_here)
    # result border?!?
    res_border: list = extract_border(result)
    # result borderless ?!?
    result_borderless: list = extract_borderless(result)
    # result cell ?!?
    res_cell: list = extract_cell(result)
    root: etree.Element = etree.Element("document")

    # if border tables detected
    if len(res_border) != 0:
        # call border script for each table in image
        for res in res_border:
            try:
                root.append(border(res, cv2.imread(image_here)))
            except:
                pass
    if len(result_borderless) != 0:
        if len(res_cell) != 0:
            for no, res in enumerate(result_borderless):
                root.append(borderless(res, cv2.imread(image_here), res_cell))
    myfile = open(xmlPath + image_here.split('/')[-1][:-3] + 'xml', "w")
    myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    myfile.write(etree.tostring(root, pretty_print=True, encoding="unicode"))
    myfile.close()


def extract_border(result) -> list:
    # for border
    res_border: list = []
    for r in result[0][0]:
        if r[4] > .85:
            res_border.append(r[:4].astype(int))
    return res_border


def extract_borderless(result) -> list:
    result_borderless = []
    # for borderless
    for r in result[0][2]:
        if r[4] > .85:
            result_borderless.append(r[:4].astype(int))
    return result_borderless


def extract_cell(result) -> list:
    result_cell = []
    # for cells
    for r in result[0][1]:
        if r[4] > .85:
            r[4] = r[4] * 100
            result_cell.append(r.astype(int))
    return result_cell


def convert_file(filepath: str) -> str:
    filename, file_extension = os.path.splitext(os.path.basename(filepath))
    if ".pdf" in file_extension:
        pages = convert_from_path(filepath, 500)
        for page in pages:
            page.save(VISUALISATION_LOCATION + "/" + filename + ".png", "PNG")
        os.remove(filepath)
        return VISUALISATION_LOCATION + "/" + filename + ".png"
    else:
        return filepath


if __name__ == "__main__":
    # List of images in the image_path
    imgs = glob.glob(image_path)
    for image in imgs:
        image = convert_file(image)
        process_image(image)
