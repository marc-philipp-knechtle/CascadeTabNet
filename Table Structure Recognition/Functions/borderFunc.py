import cv2
from Functions.line_detection import line_detection
from loguru import logger
from typing import Tuple, List, Optional


def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4) -> Tuple[int, int]:
    """

    Args:
        x1: line1, point1 x-coordinate
        y1: line1, point1 y-coordinate
        x2: line1, point2 x-coordinate
        y2: line1, point2 y-coordinate
        x3: line2, point1 x-coordinate
        y3: line2, point1 y-coordinate
        x4: line2, point2 x-coordinate
        y4: line2, point2 y-coordinate

    Returns: the intersection point (x, y) of the lines if an intersection is present, (x1, y3)

    """

    if ((x1 >= x3 - 5 or x1 >= x3 + 5) and (x1 <= x4 + 5 or x1 <= x4 - 5) and (
            y3 + 8 >= min(y1, y2) or y3 - 5 >= min(y1, y2)) and y3 <= max(y1, y2) + 5):
        return x1, y3


def extract_table(table_body, __line__, lines=None) -> List[List]:
    """
    Main extraction function
    Args:
        table_body: numpy image representation
        __line__: Decision parameter whether table is bordered or borderless. 0=borderless, 1=bordered
        lines: lines for borderless table

    Returns: Array of cells with structure:
    List[List[cell_coord_1_x, cell_coord2_y, ..., cell_coord_4_x, cell_coord_4_y]]
    Represents the following Cell-Coordinate Structure:
    [top-left, bottom-left, top-right, bottom-right]
    Returns the Cells Bounding boxes in a cell-bounding box manner.
    The bounding box is around the cell, NOT the cell content!
    """
    # Deciding variable
    if __line__ == 1:
        # Check if table image is  bordered or borderless
        logger.debug("Extracting bordered lines.")
        temp_lines_hor, temp_lines_ver = line_detection(table_body)
        logger.debug("Extracted bordered table lines.")
    else:
        temp_lines_hor, temp_lines_ver = lines

    if len(temp_lines_hor) == 0 or len(temp_lines_ver) == 0:
        logger.debug("Either Horizontal Or Vertical Lines Not Detected")
        raise RuntimeError("Cant detect bordered table without lines.")

    # List of all Rows, each row with List of Columns, each Column with List of points
    points: List[List[List]] = []
    print("[Table status] : Processing table with lines")
    logger.debug("[Table status] : Processing table with lines")

    # Remove same lines detected closer
    for x1, y1, x2, y2 in temp_lines_ver:
        point: List[List] = []
        for x3, y3, x4, y4 in temp_lines_hor:
            try:
                x, y = line_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                point.append([x, y])
            except:
                continue
        points.append(point)

    # Visualization of the detected points
    # table = table_body.copy()
    # for point in points:
    #     for x, y in point:
    #         cv2.line(table, (x, y), (x, y), (0, 0, 255), 8)
    # cv2.imshow("intersection",table)
    # cv2.waitKey(0)

    cell_bboxes: List[List] = []
    # each list elements looks like this: [cell_coord_1_x, cell_coord2_y, ..., cell_coord_4_x, cell_coord_4_y]
    cache: List[List] = []
    # creating bounding boxes of cells from the points detected
    logger.debug("Create cell bounding boxes with the detected points.")
    logger.debug("Processing detected table with " + str(len(points)) + " detected rows.")
    # This is still under work and might fail on some images
    for index_row, row in enumerate(points):
        logger.debug(
            "Processing detected row at index: " + str(index_row) + " from a total of " + str(len(points)) + " rows.")
        number_of_columns = len(row)
        next_cache: List[List] = []
        # enumeration through each column in the detected table row
        column: List[int]  # consists of List[int, int] -> each for one column position for the detected row
        for index_column, column in enumerate(row):
            logger.debug("Processing detected column at index: " + str(index_column) + " from a total of " + str(
                len(row)) + " columns.")
            if index_column == number_of_columns - 1:
                break
            # it's not possible to find horizontal neighbours in the first row -> skip
            if index_row == 0:
                next_column = row[index_column + 1]
                cache.append([column[0], column[1], next_column[0], next_column[1], None, None, None, None])
            else:
                next_column = row[index_column + 1]
                next_cache.append([column[0], column[1], next_column[0], next_column[1], None, None, None, None])
                matching_coordinates_found: bool = False
                indexes_to_remove = []
                logger.debug("Searching in cache for matching cells with size: " + str(len(cache)))
                cached_cell: List[int]  # List[cell_coord_1_x, cell_coord_2_y, ..., cell_coord_4_x, cell_coord_4_y]
                for index_k, cached_cell in enumerate(cache):

                    # column y value matches cached cell point1 y value and cache cell has no top right element
                    if (column[1] == cached_cell[1]) and cache[index_k][4] is None:
                        cache[index_k][4] = column[0]
                        cache[index_k][5] = column[1]
                        # cached cel already has bottom right element
                        if cache[index_k][4] is not None and cache[index_k][6] is not None:
                            cell_bboxes.append(cache[index_k])
                            indexes_to_remove.append(index_k)
                            matching_coordinates_found = True

                    if (next_column[1] == cached_cell[3]) and cache[index_k][6] is None:
                        cache[index_k][6] = next_column[0]
                        cache[index_k][7] = next_column[1]
                        if cache[index_k][4] is not None and cache[index_k][6] is not None:
                            cell_bboxes.append(cache[index_k])
                            indexes_to_remove.append(index_k)
                            matching_coordinates_found = True

                    # This part is not explainable for me. Why should the matching_coordinates_found flag be set to
                    # False if the last cached_cell did not match any cell?
                    # But I left this inside the file, because this is a big change compared to the original paper.
                    # if len(cache) != 0:
                    #     if cache[index_k][4] is None or cache[index_k][6] is None:
                    #         matching_coordinates_found = False
                for index_k in indexes_to_remove:
                    cache.pop(index_k)

                if not matching_coordinates_found:
                    cached_cells_removed: int = 0
                    cached_cells_appended: int = 0
                    for cached_cell in cache:
                        if cached_cell[4] is None or cached_cell[6] is None:
                            next_cache.append(cached_cell)
                            cached_cells_appended += 1
                        else:
                            cached_cells_removed += 1
                    logger.info(
                        "Did not found matching coordinates for cell -> "
                        "The current cache is added to the cache for the next row.")
                    logger.info("Removed " + str(cached_cells_removed) + " cells from the current cache.")
                    logger.info("Added " + str(cached_cells_appended) + " cells to the next cache.")

        if index_row != 0:
            logger.debug("Creating cache with current constructed cache with length: " + str(len(next_cache)))
            cache = next_cache

    # Visualizing the cells
    # table = table_body.copy()
    # count = 1
    # for i in cell_bboxes:
    #     cv2.rectangle(table_body, (i[0], i[1]), (i[6], i[7]), (int(i[7]%255),0,int(i[0]%255)), 2)
    # #     count+=1
    # cv2.imshow("cells",table_body)
    # cv2.waitKey(0)

    return cell_bboxes


# extract_table(cv2.imread("E:\\KSK\\KSK ML\\KSK PAPERS\\TabXNet\\For Git\\images\\table.PNG"),1,lines=None)


def _find_x(X, x):
    return X.index(x)


def _find_y(Y, y):
    return Y.index(y)


def span(box, X, Y):
    start_col = _find_x(X, box[0])  # x1
    end_col = _find_x(X, box[4]) - 1  # x3
    start_row = _find_y(Y, box[1])  # y1
    end_row = _find_y(Y, box[3]) - 1  # y2
    # print(end_col,end_row,start_col,start_row)
    return end_col, end_row, start_col, start_row


def extract_text_bounding_box(img) -> Optional[List[int]]:
    """
    Args:
        img: visual representation of only the cell
    Returns: the text coordinates inside the cell [x, y, width, height]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # cv2_imshow(thresh1)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=2)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    max_x, max_y, max_width, max_height = float('Inf'), float('Inf'), -1, -1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # print(im2.shape)
        if x < 2 or y < 2 or (x + w >= im2.shape[1] - 1 and y + h >= im2.shape[0] - 1) or w >= im2.shape[1] - 1 \
                or h >= im2.shape[0] - 1:
            continue
        if x < max_x:
            max_x = x
        if y < max_y:
            max_y = y
        if x + w > max_width:
            max_width = x + w
        if y + h > max_height:
            max_height = y + h

    if max_x != float('Inf') and max_y != float('Inf'):
        # Drawing a rectangle on copied image 
        # cv2.rectangle(im2, (max_x+1, max_y), (max_width-2, max_height-2), (0, 255, 0), 1)
        # cv2.imshow("detected cells", im2)
        # cv2.waitKey(0)
        return [max_x, max_y, max_width, max_height]
    else:
        return None
