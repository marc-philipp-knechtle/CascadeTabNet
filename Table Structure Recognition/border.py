import cv2
import lxml.etree as etree

from Functions.borderFunc import extract_table, extractText, span


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
    # print("x = ",x)
    # print("Y = ",Y)

    table_xml = etree.Element("table")
    t_coords = etree.Element("Coords", points=str(table[0]) + "," + str(table[1]) + " " + str(table[2]) + "," + str(
        table[3]) + " " + str(table[2]) + "," + str(table[3]) + " " + str(table[2]) + "," + str(table[1]))
    table_xml.append(t_coords)
    cv2.rectangle(imag, (table[0], table[1]), (table[2], table[3]), (0, 255, 0), 2)
    for box in final:
        if box[0] > table[0] - 5 and box[1] > table[1] - 5 and box[2] < table[2] + 5 and box[3] < table[3] + 5:
            cell_box = extractText(imag[box[1]:box[3], box[0]:box[4]])
            if cell_box is None:
                continue
            # to visualize the detected text areas
            cv2.rectangle(imag, (cell_box[0] + box[0], cell_box[1] + box[1]), (cell_box[2] + box[0], cell_box[3] + box[1]),
                          (255, 0, 0), 2)
            cell = etree.Element("cell")
            end_col, end_row, start_col, start_row = span(box, x, y)
            cell.set("end-col", str(end_col))
            cell.set("end-row", str(end_row))
            cell.set("start-col", str(start_col))
            cell.set("start-row", str(start_row))

            one = str(cell_box[0] + box[0]) + "," + str(cell_box[1] + box[1])
            two = str(cell_box[0] + box[0]) + "," + str(cell_box[3] + box[1])
            three = str(cell_box[2] + box[0]) + "," + str(cell_box[3] + box[1])
            four = str(cell_box[2] + box[0]) + "," + str(cell_box[1] + box[1])

            coords = etree.Element("Coords", points=one + " " + two + " " + three + " " + four)

            cell.append(coords)
            table_xml.append(cell)
    # to visualize the detected text areas
    # cv2.imshow("detected cells",imag)
    # cv2.waitKey(0)
    return table_xml

# border([111,228,680,480],cv2.imread('cTDaR_t10039.jpg'))
