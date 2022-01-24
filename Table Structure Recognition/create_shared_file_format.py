#!/usr/bin/env python
import glob
import os
import sys

from PIL import Image
from loguru import logger
from mmdet.apis import inference_detector, init_detector
from pdf2image import convert_from_path

from docrecjson.commontypes import Point
from docrecjson.elements import Document

SCRIPTS_LOCATION: str = "/home/makn/workspace-uni/CascadeTabNetTests"
CASCADE_TAB_NET_REPO_LOCATION: str = SCRIPTS_LOCATION + "/CascadeTabNet"
VISUALISATION_LOCATION: str = "/home/makn/Downloads/sample-tables/ba_test_tables"

# Todo these as arguments with argparse
IMAGE_PATH: str = '/home/makn/Downloads/sample-tables/2.9.scan.png'
xmlPath = '/home/makn/Downloads/sample-xml/'

config_fname = CASCADE_TAB_NET_REPO_LOCATION + "/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py"
checkpoint_file = SCRIPTS_LOCATION + "/epoch_36.pth"

model = init_detector(config_fname, checkpoint_file)

# remove the default loguru logger
logger.remove()
# add new custom loggers
logger.add(sys.stdout, level='INFO')
logger.add("failures.log", level='ERROR', rotation="10 MB")


def process_image(image_path: str):
    """

    Args:
        image_path:

    Returns:

    """
    """
    from inference_detector documentation:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    -> this version should return a generator
    """
    result = inference_detector(model, image_path)
    # result border?!?
    result_border: list = extract_border(result)
    # result borderless ?!?
    result_borderless: list = extract_borderless(result)
    # result cell ?!?
    result_cells_detection: list = extract_cell(result)

    result_cells_bounding_boxes: list = create_bounding_boxes(result_cells_detection)

    # Polygon

    image: Image = Image.open(image_path)
    # write_to_file(image_path, root)
    logger.info("Create json for [{}]", image_path)
    doc: Document = Document(version="cascade-tab-net", filename=os.path.basename(image_path),
                             original_image_size=(image.width, image.height),
                             content=[])
    logger.info("Created document from shared-file-format: \n{}", str(doc))


def create_bounding_boxes(cells: list) -> list:
    bounding_box_cells: list = []
    for cell in cells:
        bounding_box_cells.append(handle_bounding_box_cell(cell))
    return bounding_box_cells


def handle_bounding_box_cell(cell: list) -> list:
    if len(cell) != 5:
        raise ValueError("The cell array didn't fulfill the expected length. Please check whether [" + str(
            cell) + "] matches the expected requirements.")
    return create_square((cell[0], cell[1]), (cell[2], cell[3]))


def create_square(top_left: Point, bottom_right: Point) -> list:
    """
    Args:
        top_left: top left Point
        bottom_right: bottom right Point
    Returns: a new Point list with top_right and bottom left computed such that a bounding box can be computed.
    """
    top_left_x, top_left_y = top_left
    bottom_right_x, bottom_right_y = bottom_right
    box_width = bottom_right_x - top_left_x
    top_right = (top_left_x + box_width, top_left_y)
    bottom_left = (bottom_right_x - box_width, bottom_right_y)
    box_cornerstones: list = [top_left, bottom_right, top_right, bottom_left]
    return box_cornerstones


def extract_border(result) -> list:
    # for border
    res_border: list = []
    for r in result[0][0]:
        if r[4] > .85:
            res_border.append(r[:4].astype(int))
    return res_border


def extract_borderless(result) -> list:
    """
    extracts borderless masks from result
    Args:
        result:

    Returns: a list of the borderless tables. Each array describes a borderless table bounding box.
    the two coordinates in the array are the top right and bottom left coordinates of the bounding box.
    """
    result_borderless = []
    for r in result[0][2]:
        if r[4] > .85:
            # slices the threshold value of
            result_borderless.append(r[:4].astype(int))
    return result_borderless


def extract_cell(result) -> list:
    """

    Args:
        result: inference_detector result

    Returns: the array of detected cells. Each array describes a cell bounding box.
    The arrays consists normally of five elements
    1. top left x coordinate
    2. top left y coordinate
    3. bottom right x coordinate
    4. bottom right y coordinate
    5. threshold value
    """
    result_cell = []
    # for cells
    for r in result[0][1]:
        if r[4] > .85:
            # to be able to append the threshold as integer value
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


def main():
    # List of images in the image_path
    imgs = glob.glob(IMAGE_PATH)
    for image_path in imgs:
        logger.info("Received image: [{}]", image_path)
        image_path = convert_file(image_path)
        process_image(image_path)


if __name__ == "__main__":
    main()
