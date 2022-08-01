import cv2
from Functions.line_detection import line_detection
from loguru import logger


##################  Functions required for Border table Recognition #################

## Return the intersection of lines only if intersection is present ##
# Input : x1, y1, x2, y2, x3, y3, x4, y4 (1: vertical, 2: horizontal)
# Output : (x,y) Intersection point
def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # print(x1, y1, x2, y2)
    # print(x3, y3, x4, y4)

    if ((x1 >= x3 - 5 or x1 >= x3 + 5) and (x1 <= x4 + 5 or x1 <= x4 - 5) and (
            y3 + 8 >= min(y1, y2) or y3 - 5 >= min(y1, y2)) and y3 <= max(y1, y2) + 5):
        return x1, y3


def extract_table(table_body, __line__, lines=None):
    """
    Main extraction function
    Args:
        table_body: numpy image representation
        __line__: Decision parameter whether table is bordered or borderless. 0=borderless, 1=bordered
        lines: lines for borderless table

    Returns: Array of cells
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
        print("Either Horizontal Or Vertical Lines Not Detected")
        return None

    table = table_body.copy()
    k = 0
    points = []
    print("[Table status] : Processing table with lines")
    logger.debug("[Table status] : Processing table with lines")
    # Remove same lines detected closer
    for x1, y1, x2, y2 in temp_lines_ver:
        point = []
        for x3, y3, x4, y4 in temp_lines_hor:
            try:
                k += 1
                x, y = line_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                point.append([x, y])
            except:
                continue
        points.append(point)

    for point in points:
        for x, y in point:
            cv2.line(table, (x, y), (x, y), (0, 0, 255), 8)

    # cv2.imshow("intersection",table)
    # cv2.waitKey(0)

    # boxno = -1
    box = []
    last_cache = []
    # creating bounding boxes of cells from the points detected
    logger.debug("Create cell bounding boxes with the detected points.")
    # This is still under work and might fail on some images
    for i, row in enumerate(points):
        limitj = len(row)
        current_vala = []
        for j, col in enumerate(row):

            if j == limitj - 1:
                break
            if i == 0:
                nextcol = row[j + 1]
                last_cache.append([col[0], col[1], nextcol[0], nextcol[1], 9999, 9999, 9999, 9999])
            else:
                nextcol = row[j + 1]
                current_vala.append([col[0], col[1], nextcol[0], nextcol[1], 9999, 9999, 9999, 9999])
                # Matching 
                flag = 1
                index = []
                for k, last in enumerate(last_cache):

                    if (col[1] == last[1]) and last_cache[k][4] == 9999:
                        last_cache[k][4] = col[0]
                        last_cache[k][5] = col[1]
                        if last_cache[k][4] != 9999 and last_cache[k][6] != 9999:
                            box.append(last_cache[k])
                            index.append(k)
                            flag = 1

                    if (nextcol[1] == last[3]) and last_cache[k][6] == 9999:
                        last_cache[k][6] = nextcol[0]
                        last_cache[k][7] = nextcol[1]
                        if last_cache[k][4] != 9999 and last_cache[k][6] != 9999:
                            box.append(last_cache[k])
                            index.append(k)
                            flag = 1

                    if len(last_cache) != 0:
                        if last_cache[k][4] == 9999 or last_cache[k][6] == 9999:
                            flag = 0
                # print(index)
                for k in index:
                    last_cache.pop(k)
                # tranfsering
                if flag == 0:
                    for last in last_cache:
                        if last[4] == 9999 or last[6] == 9999:
                            current_vala.append(last)

        if i != 0:
            last_cache = current_vala

    # Visualizing the cells
    # count = 1
    # for i in box:
    #     cv2.rectangle(table_body, (i[0], i[1]), (i[6], i[7]), (int(i[7]%255),0,int(i[0]%255)), 2)
    # #     count+=1
    # cv2.imshow("cells",table_body)
    # cv2.waitKey(0)

    return box


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


def extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # cv2_imshow(thresh1)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=2)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    mx, my, mw, mh = float('Inf'), float('Inf'), -1, -1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # print(im2.shape)
        if x < 2 or y < 2 or (x + w >= im2.shape[1] - 1 and y + h >= im2.shape[0] - 1) or w >= im2.shape[1] - 1 or h >= \
                im2.shape[0] - 1:
            continue
        if x < mx:
            mx = x
        if y < my:
            my = y
        if x + w > mw:
            mw = x + w
        if y + h > mh:
            mh = y + h
        # print(x, y, w, h)

    if mx != float('Inf') and my != float('Inf'):
        # Drawing a rectangle on copied image 
        # rect = cv2.rectangle(im2, (mx+1, my), (mw-2, mh-2), (0, 255, 0), 1)
        # cv2_imshow(im2)
        return mx, my, mw, mh
    else:
        return None
