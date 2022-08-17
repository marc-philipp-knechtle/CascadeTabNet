import cv2
import numpy as np
from typing import List

import pytesseract

from Functions.borderFunc import extract_table

from docrecjson.elements import Document, Cell


# Input : roi of one cell
# Output : bounding box for the text in that cell
def extract_text_bless(img):
    return_arr = []
    h, w = img.shape[0:2]
    base_size = h + 14, w + 14, 3
    img_np = np.zeros(base_size, dtype=np.uint8)
    cv2.rectangle(img_np, (0, 0), (w + 14, h + 14), (255, 255, 255), 30)
    img_np[7:h + 7, 7:w + 7] = img
    # cv2_imshow(img_np)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=2)
    # cv2_imshow(dilation)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if (h < 6) or w < 4 or h / img.shape[0] > 0.95 or h > 30:
            continue
        return_arr.append([x - 7, y - 7, w, h])
    return return_arr


def handle_borderless_table(table: list, image, resolved_cells: list, document: Document) -> Document:
    cells = []
    x_lines = []
    y_lines = []
    table[0], table[1], table[2], table[3] = table[0] - 15, table[1] - 15, table[2] + 15, table[3] + 15
    for cell in resolved_cells:
        if cell[0] > table[0] - 50 and cell[1] > table[1] - 50 and cell[2] < table[2] + 50 and cell[3] < table[3] + 50:
            cells.append(cell)
            # print(cell)
    cells = sorted(cells, key=lambda x: x[3])
    row = []
    last = -1111
    row.append(table[1])
    y_lines.append([table[0], table[1], table[2], table[1]])
    temp = -1111
    prev = None
    im2 = image.copy()
    for i, cell in enumerate(cells):
        if i == 0:
            last = cell[1]
            temp = cell[3]
        elif (last + 15 > cell[1] > last - 15) or (temp + 15 > cell[3] > temp - 15):
            if cell[3] > temp:
                temp = cell[3]
        else:
            last = cell[1]
            if last > temp:
                row.append((last + temp) // 2)
            if prev is not None:
                if ((last + temp) // 2) < prev + 10 or ((last + temp) // 2) < prev - 10:
                    row.pop()
            prev = (last + temp) // 2
            temp = cell[3]

    row.append(table[3] + 50)
    i = 1
    rows = []
    for r in range(len(row)):
        rows.append([])
    final_rows = rows
    maxr = -111
    for cell in cells:
        if cell[3] < row[i]:
            rows[i - 1].append(cell)
        else:
            i += 1
            rows[i - 1].append(cell)
    for n, r1 in enumerate(rows):
        if n == len(rows):
            r1 = r1[:-1]
        # print(r1)
        r1 = sorted(r1, key=lambda x: x[0])
        prevr = None
        for no, r in enumerate(r1):
            if prevr is not None:
                if (prevr[0] + 5 >= r[0] >= prevr[0] - 5) or (prevr[2] + 5 >= r[2] >= prevr[2] - 5):
                    if r[4] < prevr[4]:
                        r1.pop(no)
                    else:
                        r1.pop(no - 1)
            prevr = r

        final_rows[n] = r1
    lasty = []
    for x in range(len(final_rows)):
        lasty.append([99999999, 0])

    prev = None
    for n, r1 in enumerate(final_rows):
        for r in r1:
            if prev is None:
                prev = r
            else:
                if r[1] < prev[3]:
                    continue

            if r[1] < lasty[n][0]:
                lasty[n][0] = r[1]
            if r[3] > lasty[n][1]:
                lasty[n][1] = r[3]
    row = [table[1]]
    prev = None
    pr = None
    for x in range(len(lasty) - 1):
        if x == 0 and prev is None:
            prev = lasty[x]
        else:
            if pr is not None:
                if abs(((lasty[x][0] + prev[1]) // 2) - pr) <= 10:
                    row.pop()
                    row.append((lasty[x][0] + prev[1]) // 2)
                else:
                    row.append((lasty[x][0] + prev[1]) // 2)
            else:
                row.append((lasty[x][0] + prev[1]) // 2)
            pr = (lasty[x][0] + prev[1]) // 2
            prev = lasty[x]
    row.append(table[3])
    maxr = 0
    for r2 in final_rows:
        print(r2)
        if len(r2) > maxr:
            maxr = len(r2)

    lastx = []

    for n in range(maxr):
        lastx.append([999999999, 0])

    for r2 in final_rows:
        if len(r2) == maxr:
            for n, col in enumerate(r2):
                # print(col)
                if col[2] > lastx[n][1]:
                    lastx[n][1] = col[2]
                if col[0] < lastx[n][0]:
                    lastx[n][0] = col[0]

    for r2 in final_rows:
        if len(r2) != 0:
            r = 0
            for n, col in enumerate(r2):
                while r != len(r2) - 1 and (lastx[n][0] > r2[r][0]):
                    r += 1
                if n != 0:
                    if r2[r - 1][0] > lastx[n - 1][1]:
                        if r2[r - 1][0] < lastx[n][0]:
                            lastx[n][0] = r2[r - 1][0]
    for r2 in final_rows:
        for n, col in enumerate(r2):
            if n != len(r2) - 1:
                if col[2] < lastx[n + 1][0]:
                    if col[2] > lastx[n][1]:
                        lastx[n][1] = col[2]

    col = np.zeros(maxr + 1)
    col[0] = table[0]
    prev = 0
    i = 1
    for x in range(len(lastx)):
        if x == 0:
            prev = lastx[x]
        else:
            col[i] = (lastx[x][0] + prev[1]) // 2
            i += 1
            prev = lastx[x]
    col = col.astype(int)
    col[maxr] = table[2]

    _row_ = sorted(row, key=lambda x: x)
    _col_ = sorted(col, key=lambda x: x)

    for no, c in enumerate(_col_):
        x_lines.append([c, table[1], c, table[3]])
        cv2.line(im2, (c, table[1]), (c, table[3]), (255, 0, 0), 1)
    for no, c in enumerate(_row_):
        y_lines.append([table[0], c, table[2], c])
        cv2.line(im2, (table[0], c), (table[2], c), (255, 0, 0), 1)

    # cv2_imshow(im2)
    # for r in row:
    #   cv2.line(im2,(r,table[1]),(r,table[3]),(0,255,0),1)
    # for c in col:
    #   cv2.line(im2,(c,table[1]),(c,table[3]),(0,255,0),1)
    final = extract_table(image[table[1]:table[3], table[0]:table[2]], 0, (y_lines, x_lines))

    cellBoxes = []
    img4 = image.copy()
    for box in final:
        cellBox = extract_text_bless(image[box[1]:box[3], box[0]:box[4]])
        for cell in cellBox:
            cellBoxes.append([box[0] + cell[0], box[1] + cell[1], cell[2], cell[3]])
            cv2.rectangle(img4, (box[0] + cell[0], box[1] + cell[1]),
                          (box[0] + cell[0] + cell[2], box[1] + cell[1] + cell[3]), (255, 0, 0), 2)
    # cv2_imshow(img4)

    the_last_y = -1
    cellBoxes = sorted(cellBoxes, key=lambda x: x[1])
    cellBoxes2BeMerged = []
    cellBoxes2BeMerged.append([])
    row_count = 0
    for cell in cellBoxes:
        if (the_last_y == -1):
            the_last_y = cell[1]
            cellBoxes2BeMerged[row_count].append(cell)
            continue
        if (abs(cell[1] - the_last_y) < 8):
            cellBoxes2BeMerged[row_count].append(cell)
        else:
            the_last_y = cell[1]
            row_count += 1
            cellBoxes2BeMerged.append([])
            cellBoxes2BeMerged[row_count].append(cell)

    merged_boxes = []
    for cellrow in cellBoxes2BeMerged:
        cellrow = sorted(cellrow, key=lambda x: x[0])
        cur_cell = -1
        for c, cell in enumerate(cellrow):
            if cur_cell == -1:
                cur_cell = cell
                continue
            if len(cellrow) == 1:
                merged_boxes.append(cell)
                break
            if abs((cur_cell[0] + cur_cell[2]) - cell[0]) < 10:
                cur_cell[2] = cur_cell[2] + cell[2] + (cell[0] - (cur_cell[0] + cur_cell[2]))
                if cur_cell[3] < cell[3]:
                    cur_cell[3] = cell[3]
            else:
                cur_cell[2] = cur_cell[0] + cur_cell[2]
                cur_cell[3] = cur_cell[1] + cur_cell[3]
                merged_boxes.append(cur_cell)
                cur_cell = cell
        if cur_cell != -1:
            cur_cell[2] = cur_cell[0] + cur_cell[2]
            cur_cell[3] = cur_cell[1] + cur_cell[3]
            merged_boxes.append(cur_cell)

    im3 = image.copy()
    for bx in merged_boxes:
        cv2.rectangle(im3, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 2)
    # cv2_imshow(im3)
    text_chunks = []
    text_chunks.append([])
    rcnt = 0
    ycnt = -1

    final = sorted(final, key=lambda x: x[1])
    for box in final:
        if ycnt == -1:
            ycnt = box[1]
        tcurcell = []
        mcurcell = []
        for mbox in merged_boxes:
            if mbox[0] >= box[0] and mbox[1] >= box[1] and mbox[2] <= box[4] and mbox[3] <= box[3]:
                if len(tcurcell) == 0:
                    tcurcell = mbox
                else:
                    if mbox[0] < tcurcell[0]:
                        tcurcell[0] = mbox[0]
                    if mbox[1] < tcurcell[1]:
                        tcurcell[1] = mbox[1]
                    if mbox[2] > tcurcell[2]:
                        tcurcell[2] = mbox[2]
                    if mbox[3] > tcurcell[3]:
                        tcurcell[3] = mbox[3]

        for i, frow in enumerate(final_rows):
            for j, fbox in enumerate(frow):
                if fbox[0] >= box[0] and fbox[0] <= box[4] and fbox[1] >= box[1] and fbox[1] <= box[3]:
                    mcurcell = fbox
                    final_rows[i].pop(j)
                    break

        if abs(ycnt - box[1]) > 10:
            rcnt += 1
            text_chunks.append([])
            ycnt = box[1]

        if len(tcurcell) == 0:
            if len(mcurcell) == 0:
                continue
            else:
                text_chunks[rcnt].append(mcurcell)
        else:
            if len(mcurcell) == 0:
                text_chunks[rcnt].append(tcurcell)
            else:
                if (abs(mcurcell[0] - tcurcell[0]) <= 20 and abs(mcurcell[1] - tcurcell[1]) <= 20 and abs(
                        mcurcell[2] - tcurcell[2]) <= 20 and abs(mcurcell[3] - tcurcell[3]) <= 20):
                    text_chunks[rcnt].append(tcurcell)
                elif ((abs(mcurcell[0] - tcurcell[0]) <= 20 and abs(mcurcell[2] - tcurcell[2]) <= 20) or (
                        abs(mcurcell[1] - tcurcell[1]) <= 20 or abs(mcurcell[3] - tcurcell[3]) <= 20)):
                    text_chunks[rcnt].append(mcurcell)
                else:
                    text_chunks[rcnt].append(tcurcell)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (125, 125, 0), (0, 255, 255)]
    for no, r in enumerate(text_chunks):
        for tbox in r:
            cv2.rectangle(im2, (tbox[0], tbox[1]), (tbox[2], tbox[3]), colors[no % len(colors)], 1)
            # print(tbox)

    # cv2.imshow("text chunks", im2)
    # cv2.waitKey(0)

    def rowstart(val):
        r = 0
        while r < len(_row_) and val > _row_[r]:
            r += 1
        if r - 1 == -1:
            return r
        else:
            return r - 1

    def rowend(val):
        r = 0
        while r < len(_row_) and val > _row_[r]:
            r += 1
        if r - 1 == -1:
            return r
        else:
            return r - 1

    def colstart(val):
        r = 0
        while r < len(_col_) and val > _col_[r]:
            r += 1
        if r - 1 == -1:
            return r
        else:
            return r - 1

    def colend(val):
        r = 0
        while r < len(_col_) and val > _col_[r]:
            r += 1
        if r - 1 == -1:
            return r
        else:
            return r - 1

    cells: List[Cell] = []
    for final in text_chunks:
        for box in final:
            end_col, end_row, start_col, start_row = colend(box[2]), rowend(box[3]), colstart(box[0]), rowstart(box[1])

            # todo add pyteseract preprocessing?
            # https://github.com/NanoNets/ocr-with-tesseract/blob/master/tesseract-tutorial.ipynb
            cv2_roi = image[box[1]:box[3], box[0]: box[2]]
            text: str = pytesseract.image_to_string(cv2_roi)

            cells.append(document.add_cell([(box[0], box[1]),
                                            (box[0], box[3]),
                                            (box[2], box[3]),
                                            (box[2], box[1])], start_row, end_row, start_col, end_col, text=text,
                                           source='prediction'))

    table = document.add_table([(table[0], table[1]), (table[0], table[3]), (table[2], table[3]), (table[2], table[1])],
                               cells, source='prediction')
    casctabnet_metadata: dict = {"CascadeTabNet Border": {"bordered": "False", "borderless": "True"}}
    document.add_content_metadata(casctabnet_metadata, group_ref=table, parent_ref=table.oid)

    return document
