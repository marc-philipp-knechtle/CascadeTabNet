import cv2
import lxml.etree as etree
from typing import List

from pytesseract import pytesseract
from shapely import geometry
from shapely.geometry import Polygon

from Functions.borderFunc import extract_table, extract_text_bounding_box, span
from docrecjson.elements import Document, Cell


# Input : table coordinates [x1,y1,x2,y2]
# Output : XML Structure for ICDAR 19 single table
def border(table, image):
    image_np = image  # [table[1]-10:table[3]+10,table[0]-10:table[2]+10]
    imag = image.copy()
    final = extract_table(image_np, 1)
    if final is None:
        return None
    x = []
    y = []
    for x1, y1, x2, y2, x3, y3, x4, y4 in final:
        if x1 not in x:
            x.append(x1)
        if x3 not in x:
            x.append(x3)
        if y1 not in y:
            y.append(y1)
        if y2 not in y:
            y.append(y2)

    x.sort()
    y.sort()

    table_xml = etree.Element("table")
    t_coords = etree.Element("Coords", points=str(table[0]) + "," + str(table[1]) + " " + str(table[2]) + "," + str(
        table[3]) + " " + str(table[2]) + "," + str(table[3]) + " " + str(table[2]) + "," + str(table[1]))
    table_xml.append(t_coords)
    cv2.rectangle(imag, (table[0], table[1]), (table[2], table[3]), (0, 255, 0), 2)
    for box in final:
        if box[0] > table[0] - 5 and box[1] > table[1] - 5 and box[2] < table[2] + 5 and box[3] < table[3] + 5:
            cell_text_bbox = extract_text_bounding_box(imag[box[1]:box[3], box[0]:box[4]])
            if cell_text_bbox is None:
                continue
            # to visualize the detected text areas
            # cv2.rectangle(imag, (cell_text_bbox[0] + box[0], cell_text_bbox[1] + box[1]),
            #               (cell_text_bbox[2] + box[0], cell_text_bbox[3] + box[1]),
            #               (255, 0, 0), 2)

            cell = etree.Element("cell")
            end_col, end_row, start_col, start_row = span(box, x, y)
            cell.set("end-col", str(end_col))
            cell.set("end-row", str(end_row))
            cell.set("start-col", str(start_col))
            cell.set("start-row", str(start_row))

            one = str(cell_text_bbox[0] + box[0]) + "," + str(cell_text_bbox[1] + box[1])
            two = str(cell_text_bbox[0] + box[0]) + "," + str(cell_text_bbox[3] + box[1])
            three = str(cell_text_bbox[2] + box[0]) + "," + str(cell_text_bbox[3] + box[1])
            four = str(cell_text_bbox[2] + box[0]) + "," + str(cell_text_bbox[1] + box[1])

            coords = etree.Element("Coords", points=one + " " + two + " " + three + " " + four)

            cell.append(coords)
            table_xml.append(cell)
    # to visualize the detected text areas
    # cv2.imshow("detected cells",imag)
    # cv2.waitKey(0)
    return table_xml


def handle_bordered_table(table: list, image, document: Document) -> Document:
    """
    Args:
        table: table coordinates representation
        image: image files, read with cv2.imread
        document: shared-file-document to add the table to.

    Returns: document with added Table
    """
    image_np = image  # [table[1]-10:table[3]+10,table[0]-10:table[2]+10]
    image_copy = image.copy()
    # Contains the detected cell coordinates
    # [cell_coord_1_x, cell_coord_1_y, ..., cell_coord_4_x, cell_coord_4_y], [...]
    final: List[List] = extract_table(image_np, 1)
    if final is None:
        raise RuntimeError(
            "Couldn't extract table from document with table detected in it. Returned None. Please review.")
    x = []
    y = []
    for x1, y1, x2, y2, x3, y3, x4, y4 in final:
        if x1 not in x:
            x.append(x1)
        if x3 not in x:
            x.append(x3)
        if y1 not in y:
            y.append(y1)
        if y2 not in y:
            y.append(y2)

    x.sort()
    y.sort()

    cells: List[Cell] = []
    cv2.rectangle(image_copy, (table[0], table[1]), (table[2], table[3]), (0, 255, 0), 2)
    for cell_bbox in final:
        if cell_bbox[0] > table[0] - 5 and cell_bbox[1] > table[1] - 5 and cell_bbox[2] < table[2] + 5 \
                and cell_bbox[3] < table[3] + 5:
            cell_text_bbox = extract_text_bounding_box(image_copy[cell_bbox[1]:cell_bbox[3], cell_bbox[0]:cell_bbox[4]])
            if cell_text_bbox is None:
                continue
            # to visualize the detected text areas
            # cv2.rectangle(image_copy, (cell_text_bbox[0] + cell_bbox[0], cell_text_bbox[1] + cell_bbox[1]),
            #               (cell_text_bbox[2] + cell_bbox[0], cell_text_bbox[3] + cell_bbox[1]),
            #               (255, 0, 0), 2)

            end_col, end_row, start_col, start_row = span(cell_bbox, x, y)
            # the handling with shapely is necessary because sometimes the coordinates are out of order
            text_cell_points: List[geometry.Point] = [
                geometry.Point(cell_text_bbox[0] + cell_bbox[0], cell_text_bbox[1] + cell_bbox[1]),
                geometry.Point(cell_text_bbox[0] + cell_bbox[0], cell_text_bbox[3] + cell_bbox[1]),
                geometry.Point(cell_text_bbox[2] + cell_bbox[0], cell_text_bbox[3] + cell_bbox[1]),
                geometry.Point(cell_text_bbox[2] + cell_bbox[0], cell_text_bbox[1] + cell_bbox[1])]
            cell_points: List[geometry.Point] = [geometry.Point(cell_bbox[0], cell_bbox[1]),
                                                 geometry.Point(cell_bbox[2], cell_bbox[3]),
                                                 geometry.Point(cell_bbox[4], cell_bbox[5]),
                                                 geometry.Point(cell_bbox[6], cell_bbox[7])]

            # the points from cell_points are used instead of text_cell_points because these are so tightly around
            # the edges of the cell.
            cell_polygon: Polygon = Polygon([p.x, p.y] for p in cell_points)
            # [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
            bounds: list = [int(x) for x in cell_polygon.bounds]  # there is no float pixel value
            # todo add pytesseract preprocessing?
            # https://github.com/NanoNets/ocr-with-tesseract/blob/master/tesseract-tutorial.ipynb
            # [y1:y2, x1:x2]
            cv2_roi = image[bounds[1]:bounds[3], bounds[0]: bounds[2]]

            text: str = pytesseract.image_to_string(cv2_roi)

            cells.append(
                document.add_cell([(int(point.x), int(point.y)) for point in text_cell_points], start_row, end_row,
                                  start_col, end_col, source='prediction', text=text))

    table = document.add_table([(table[0], table[1]), (table[0], table[2]), (table[2], table[3]), (table[2], table[1])],
                               cells)
    casctabnet_metadata: dict = {"CascadeTabNet Border": {"bordered": "True", "borderless": "False"}}
    document.add_content_metadata(casctabnet_metadata, group_ref=table, parent_ref=table.oid)
    # to visualize the detected text areas
    # cv2.imshow("detected cells",image_copy)
    # cv2.waitKey(0)
    return document
