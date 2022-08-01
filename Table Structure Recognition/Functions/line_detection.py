import cv2
import numpy as np
from loguru import logger


# Input : Image
# Output : hor,ver 
def line_detection(image):
    """
    Args:
        image: numpy image

    Returns: horizontal and vertical lines
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
    bw = cv2.bitwise_not(bw)

    # To visualize image after thresholding ##
    # cv2.imshow("bw",bw)
    # cv2.waitKey(0)

    horizontal = bw.copy()
    vertical = bw.copy()
    # [horizontal lines]
    # Create structure element for extracting horizontal lines through morphology operations
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    horizontal = cv2.dilate(horizontal, (1, 1), iterations=5)
    horizontal = cv2.erode(horizontal, (1, 1), iterations=5)

    # Uncomment to visualize highlighted Horizontal lines
    # cv2.imshow("horizontal",horizontal)
    # cv2.waitKey(0)

    # HoughlinesP function to detect horizontal lines
    logger.debug("Executing HoughLinesP horizontal line detection.")
    horizontal_lines = cv2.HoughLinesP(horizontal, rho=1, theta=np.pi / 180, threshold=100, minLineLength=30,
                                       maxLineGap=3)
    if horizontal_lines is None:
        return None, None
    temp_line = []
    for line in horizontal_lines:
        for x1, y1, x2, y2 in line:
            temp_line.append([x1, y1 - 5, x2, y2 - 5])

    # Sorting the list of detected lines by Y1
    horizontal_lines = sorted(temp_line, key=lambda x: x[1])

    # Uncomment this part to visualize the lines detected on the image #
    # print(len(horizontal_lines))
    # for x1, y1, x2, y2 in horizontal_lines:
    #     cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)

    # print(image.shape)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)

    # Selection of best lines from all the horizontal lines detected
    lasty1 = -111111
    lines_x1 = []
    lines_x2 = []
    hor = []
    i = 0
    logger.debug("Selecting best horizontal lines.")
    for x1, y1, x2, y2 in horizontal_lines:
        if lasty1 <= y1 <= lasty1 + 10:
            lines_x1.append(x1)
            lines_x2.append(x2)
        else:
            if i != 0 and len(lines_x1) is not 0:
                hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
            lasty1 = y1
            lines_x1 = []
            lines_x2 = []
            lines_x1.append(x1)
            lines_x2.append(x2)
            i += 1
    hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
    #####################################################################

    # [vertical lines]
    # Create structure element for extracting vertical lines through morphology operations
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    # Apply morphology operations
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    vertical = cv2.dilate(vertical, (1, 1), iterations=8)
    vertical = cv2.erode(vertical, (1, 1), iterations=7)

    # Preprocessing Vertical Lines
    # cv2.imshow("vertical",vertical)
    # cv2.waitKey(0)

    # HoughlinesP function to detect vertical lines
    logger.debug("Executing HoughLinesP vertical line detection.")
    # ver_lines = cv2.HoughLinesP(vertical,rho=1,theta=np.pi/180,threshold=20,minLineLength=20,maxLineGap=2)
    ver_lines = cv2.HoughLinesP(vertical, 1, np.pi / 180, 20, np.array([]), 20, 2)
    if ver_lines is None:
        return None, None
    temp_line = []
    for line in ver_lines:
        for x1, y1, x2, y2 in line:
            temp_line.append([x1, y1, x2, y2])

    # Sorting the list of detected lines by X1
    ver_lines = sorted(temp_line, key=lambda x: x[0])

    # Uncomment this part to visualize the lines detected on the image
    # print(len(ver_lines))
    # for x1, y1, x2, y2 in ver_lines:
    #     cv2.line(image, (x1,y1-5), (x2,y2-5), (0, 255, 0), 1)

    # print(image.shape)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)

    logger.debug("Selecting best vertical lines.")
    lastx1 = -111111
    lines_y1 = []
    lines_y2 = []
    ver = []
    count = 0
    lasty1 = -11111
    lasty2 = -11111
    for x1, y1, x2, y2 in ver_lines:
        if lastx1 <= x1 <= lastx1 + 15 and not (
                (min(y1, y2) < min(lasty1, lasty2) - 20 or min(y1, y2) < min(lasty1, lasty2) + 20) and (
                (max(y1, y2) < max(lasty1, lasty2) - 20 or max(y1, y2) < max(lasty1, lasty2) + 20))):
            lines_y1.append(y1)
            lines_y2.append(y2)
            # lasty1 = y1
            # lasty2 = y2
        else:
            if count != 0 and len(lines_y1) is not 0:
                ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])
            lastx1 = x1
            lines_y1 = []
            lines_y2 = []
            lines_y1.append(y1)
            lines_y2.append(y2)
            count += 1
            lasty1 = -11111
            lasty2 = -11111
    ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])

    # Visualization of Lines After Post Processing
    # for x1, y1, x2, y2 in ver:
    #     cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)

    # for x1, y1, x2, y2 in hor:
    #     cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)

    # cv2.imshow("image",img)
    # cv2.waitKey(0)

    return hor, ver

# line_detection(cv2.imread('path to image'))
