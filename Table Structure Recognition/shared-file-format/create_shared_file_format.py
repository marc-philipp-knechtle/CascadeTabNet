#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from typing import List, Tuple

from PIL import Image
from loguru import logger
from mmdet.apis import inference_detector, init_detector
from pdf2image import convert_from_path

from database.db import Connection
from docrecjson.commontypes import Point
from docrecjson.elements import Document, PolygonRegion, Cell

# remove the default loguru logger
logger.remove()
# add new custom loggers
logger.add(sys.stdout, level='INFO')
logger.add("info.log", level='INFO', rotation="10 MB")
logger.add("failures.log", level='ERROR', rotation="10 MB")

__db = Connection()


def process_image(image_path: str, config_fname: str, checkpoint_file: str) -> Document:
    """
    from inference_detector documentation:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    -> this version should return a generator
    """
    model = init_detector(config_fname, checkpoint_file)
    result = inference_detector(model, image_path)
    # result = None

    # These may be included in the shared-file-format results, but I'm unsure whether this is applicable.
    # result border?!?
    # result_border: list = extract_border(result)
    # result borderless ?!?
    # result_borderless: list = extract_borderless(result)
    # result cell ?!?
    result_cells_detection: list = extract_cell(result)

    result_cells_bounding_boxes: List[List[Point]] = create_bounding_boxes(result_cells_detection)

    image: Image = Image.open(image_path)
    # write_to_file(image_path, root)
    logger.info("Create json for [{}]", image_path)
    doc: Document = Document.empty(filename=os.path.basename(image_path),
                                   original_image_size=(image.width, image.height))
    doc.set_source_for_adding("prediction")
    doc.add_creator("CascadeTabNet", "1.0")

    cells: List[Cell] = []
    for cell in result_cells_bounding_boxes:
        # the cell array has a weird format which produces conflicts with other applications in downstream tasks
        # they produce a cross-like shape for detection
        # this is the reason the cell list is reordered properly
        cell_ordered: list = [cell[0], cell[2], cell[1], cell[3]]
        cell: Cell = doc.add_cell(cell_ordered, source='prediction')
        cells.append(cell)

    if len(cells) != 0:
        doc.add_table(get_table_coordinates_from_cells(cells), cells, source="prediction")

    logger.info("Created document from shared-file-format: \n{}", str(doc.to_json()))
    logger.info("Finished shared-file-format creation on: \n{}", image_path)
    logger.info("Waiting for new files...")

    return doc


def get_table_coordinates_from_cells(cells: List[Cell]) -> list:
    """
    Computes the cell bounding box based on the already extracted cells.
    It just needs to compute the lower left coordinate, as well as the upper right coordinate.
    The remaining coordinates can be computed with _span_polygon
    :param cells: all cells in the tables. The cell's bounding box can be accessed via cell.bounding_box.
                  The single coordinates are in the order as they are returned by _span_polygon.
                  This is because _span_polygon was already used for the cell bounding box creation.
    :return: all four rectangle coordinates of the table bounding box
    """
    all_x_values = []
    all_y_values = []

    for cell in cells:
        for point in cell.bounding_box.polygon:
            all_x_values.append(point[0])
            all_y_values.append(point[1])

    # lower left coordinate = min x coordinate + max y coordinate
    # upper right coordinate = max x coordinate + min y coordinate
    max_x = max(all_x_values)
    min_x = min(all_x_values)
    max_y = max(all_y_values)
    min_y = min(all_y_values)

    return _span_polygon((min_x, max_y), (max_x, min_y))


def _span_polygon(point1: Tuple, point2: Tuple) -> list:
    """
    The sci tsr polygon bounding boxes do not have the necessary coordinate structure for the shared-file-format.
    The coordinates are simply the lower left of the bbox and the upper right of the bbox.
    But this is sufficient to construct the right square coordinates.
    It's important that the coordinates are in the right order because the shared-file-format assumes that the last
    coordinates are connected.
    :param point1:  lower left coordinate
    :param point2: upper right coordinate
    :return: a list of four coordinates. with index:
        0 = lower left
        1 = lower right
        2 = upper right
        3 = upper left
    """
    # written not in a single statement for readability
    # noinspection PyListCreation
    polygon_points: list = []

    polygon_points.append(point1)
    # point2.x, point1.y = lower right
    polygon_points.append((point2[0], point1[1]))
    polygon_points.append(point2)
    # point1.x, point2.y = upper left
    polygon_points.append((point1[0], point2[1]))

    return polygon_points


def create_bounding_boxes(cells: list) -> List[List[Point]]:
    bounding_box_cells: list = []
    for cell in cells:
        bounding_box_cells.append(handle_bounding_box_cell(cell))
    return bounding_box_cells


def handle_bounding_box_cell(cell: list) -> List[Point]:
    if len(cell) != 5:
        raise ValueError("The cell array didn't fulfill the expected length. Please check whether [" + str(
            cell) + "] matches the expected requirements.")
    return create_square((cell[0], cell[1]), (cell[2], cell[3]))


def create_square(top_left: Point, bottom_right: Point) -> List[Point]:
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
    """
    Args:
        filepath: file to check and convert if necessary
    Returns: image file if pdf was specified
    """
    filename, file_extension = os.path.splitext(os.path.basename(filepath))
    dir: str = os.path.dirname(filepath)
    if ".pdf" in file_extension:
        pages = convert_from_path(filepath, 500)
        for page in pages:
            page.save(dir + "/" + filename + ".png", "PNG")
        os.remove(filepath)
        return dir + "/" + filename + ".png"
    else:
        return filepath


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--checkpoint",
                        help="Please add the path to your downloaded, pretrained mode."
                             "The default assumes that it's currently in your working directory.",
                        default="epoch_36.pth",
                        type=str)
    parser.add_argument("-co", "--config",
                        help="Please add the path to your config file. This file resides in the CascadeTabNet/Config"
                             "directory.",
                        default=7, type=str)
    parser.add_argument("-e", "--extraction", help="Folder to monitor for new incoming extraction files.",
                        type=str)
    parser.add_argument("-ed", "--extractionDetected",
                        help="Folder to move extracted, successfully detected files to.",
                        type=str)
    parser.add_argument("-ej", "--extractionJson",
                        help="Specify a folder to save the extracted json Files into. "
                             "Leave empty, if you don't want to save any shared-file-format json files",
                        type=str, default="")

    return parser.parse_args()


def handle_duplicate_files(filepath: str, new_folder_location: str):
    """
    handles duplicate files + adds e.g. filenameXYZ(1).jpg counter behind it.
    And SAVES IT!

    Args:
        filepath: filepath which was tried to be written
        new_folder_location: the folder where the file should be in

    Returns: nothing

    """
    counter: int = 1
    filename, file_extension = os.path.splitext(os.path.basename(filepath))
    while os.path.isfile(os.path.join(new_folder_location, filename + " (" + str(counter) + ")" + file_extension)):
        counter += 1
    os.rename(filepath, os.path.join(new_folder_location, filename + " (" + str(counter) + ")" + file_extension))


def move_to_folder(filepath: str, new_folder_location: str):
    # todo maybe remove os.path.join... because filepath is already path
    if not os.path.isfile(filepath):
        os.rename(filepath, new_folder_location)
    else:
        handle_duplicate_files(filepath, new_folder_location)


def save_as_json(shared_file_document: Document, filepath: str):
    with open(filepath + ".json", 'w') as json_file:
        json.dump(shared_file_document.to_dict(), json_file)
        logger.info("Saved json file: " + filepath + ".json")


def main(checkpoint_filepath: str, config_filepath: str, extraction_filepath: str, extraction_detected_filepath: str,
         extraction_json_filepath: str):
    logger.info("Waiting for new files...")
    try:
        while True:
            for filename in os.listdir(extraction_filepath):
                filepath = extraction_filepath + "/" + filename
                logger.info("Received image: [{}]", filepath)
                image_path: str = convert_file(filepath)
                extracted_image: Document = process_image(image_path, config_filepath, checkpoint_filepath)
                __db.get_collection().insert_one(extracted_image.to_dict())
                move_to_folder(image_path, extraction_detected_filepath)

                # extract pure filename from this path
                filename_without_extension = filename.rsplit('.', maxsplit=1)[0]
                save_as_json(extracted_image, os.path.join(extraction_json_filepath, filename_without_extension))
            time.sleep(2)
    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args.checkpoint, args.config, args.extraction, args.extractionDetected, args.extractionJson)
