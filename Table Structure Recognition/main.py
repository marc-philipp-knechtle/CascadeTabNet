#!/usr/bin/env python
import glob

import cv2
import lxml.etree as etree
from mmdet.apis import inference_detector, init_detector

from Functions.blessFunc import borderless
from border import border

SCRIPTS_LOCATION: str = "/home/makn/workspace-uni/CascadeTabNetTests"
CASCADE_TAB_NET_REPO_LOCATION: str = SCRIPTS_LOCATION + "/CascadeTabNet"

# To Do
image_path = '/home/makn/Downloads/sample-tables/invoice.jpg'
xmlPath = '/home/makn/Downloads/sample-xml/'

config_fname = CASCADE_TAB_NET_REPO_LOCATION + "/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py"
checkpoint_file = SCRIPTS_LOCATION + "/epoch_36.pth"

model = init_detector(config_fname, checkpoint_file)

if __name__ == "__main__":
    # List of images in the image_path
    imgs = glob.glob(image_path)
    for i in imgs:
        result = inference_detector(model, i)
        res_border = []
        res_bless = []
        res_cell = []
        root = etree.Element("document")
        # for border
        for r in result[0][0]:
            if r[4] > .85:
                res_border.append(r[:4].astype(int))
        # for cells
        for r in result[0][1]:
            if r[4] > .85:
                r[4] = r[4] * 100
                res_cell.append(r.astype(int))
        # for borderless
        for r in result[0][2]:
            if r[4] > .85:
                res_bless.append(r[:4].astype(int))

        # if border tables detected
        if len(res_border) != 0:
            # call border script for each table in image
            for res in res_border:
                try:
                    root.append(border(res, cv2.imread(i)))
                except:
                    pass
        if len(res_bless) != 0:
            if len(res_cell) != 0:
                for no, res in enumerate(res_bless):
                    root.append(borderless(res, cv2.imread(i), res_cell))

        myfile = open(xmlPath + i.split('/')[-1][:-3] + 'xml', "w")
        myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        myfile.write(etree.tostring(root, pretty_print=True, encoding="unicode"))
        myfile.close()
