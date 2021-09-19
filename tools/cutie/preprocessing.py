from tools.yolo.preprocessing import clean_via_file
from tools.image.imageTools import get_img_shape
import cv2
import numpy as np
import pandas as pd
import os

def divide_boxes(via_file: str) -> dict:
    """Divides the boxes containing several words in the via annotations file given as parameter.
    Returns a dictionary with the information of each box : filename, class, word, x, y, width, height.

    :param via_file: path to the via csv file
    :type via_file: str
    :return: dictionary with the boxes information
    :rtype: dict
    """
    via_clean = clean_via_file(via_file) # clean the initial file to get the right separators
    images_boxes = pd.read_csv(via_clean,sep='|',header=0)
    images_boxes = images_boxes[['filename','region_shape_attributes','region_attributes']]
    images_boxes['box_shape'] = images_boxes.region_shape_attributes.str.strip('"').str\
                                    .replace('""""','"')
    images_boxes['attributes'] =  images_boxes.region_attributes.str.strip('"').str\
                                    .replace('""""','"')
    images_boxes['text'] = ""
    images_boxes['class'] = ""
    
    for index, attribute in images_boxes['attributes'].iteritems():
        images_boxes.at[index,'text'] = eval(attribute)['Text']
        images_boxes.at[index,'class'] = int(eval(attribute)['type'])

    images_boxes.drop(columns=['region_shape_attributes', 'region_attributes' \
           , 'attributes'],inplace=True)

    final_dict = {'filename':[],'class':[],'word':[],'x':[],'y':[],'width':[],'height':[]}

    for index, text in images_boxes['text'].iteritems():
        total_words = len(text.split(' '))
        total_chars = len(text.replace(' ',''))
        all_points_x = eval(images_boxes.at[index,'box_shape'])['all_points_x']
        all_points_y = eval(images_boxes.at[index,'box_shape'])['all_points_y']
        x_min = min(all_points_x)
        x_max = max(all_points_x)
        y_min = min(all_points_y)
        y_max = max(all_points_y)
        box_width = x_max - x_min
        box_height = y_max - y_min
        char_counter = 0
        for word_idx in range(total_words):
            word_chars = len(text.split(' ')[word_idx])
            x_min_word = x_min + ((word_idx + char_counter) * box_width / total_chars)
            x_max_word = x_min + ((word_idx + char_counter + word_chars) * box_width / total_chars)
            char_counter += word_chars

            final_dict['filename'].append(images_boxes.at[index,'filename'])
            final_dict['class'].append(images_boxes.at[index,'class'])
            final_dict['word'].append(text.split(' ')[word_idx])
            final_dict['x'].append((x_min_word + x_max_word)/2)
            final_dict['y'].append((y_min + y_max)/2)
            final_dict['width'].append((x_max_word - x_min_word))
            final_dict['height'].append(box_height)

    return(final_dict)

def check_box_dictionary(image_path: str, box_dict: dict):
    """Opens a new window showing the image and the boxes found in the dictionary given as parameter.
    
    :param image_path: path to the image
    :type image_path: str
    :param box_dict: dictionary prepared with divide_boxes function
    :type box_dict: dict
    """
    filename = os.path.basename(image_path)
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (500,500), interpolation = cv2.INTER_AREA)
    im_height, im_width, _ = get_img_shape(image_path)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    fontScale = 0.4
    color = (255, 0, 0)
    thickness = 1

    for index in range(len(box_dict['filename'])):
        if box_dict['filename'][index] == filename:
            rect_class = box_dict['class'][index]
            x = float((box_dict['x'][index])/im_width)
            y = float((box_dict['y'][index])/im_height)
            width = (box_dict['width'][index]/im_width)
            height = (box_dict['height'][index]/im_height)
            x1 = int((x - (width / 2)) * 500)
            y1 = int((y - (height / 2)) * 500)
            x2 = int((x + (width / 2)) * 500)
            y2 = int((y + (height / 2)) * 500)
            color = ((rect_class==1)*50 + (rect_class==2)*255, (rect_class==1)*150 + (rect_class==3)*255,(rect_class==1)*255 + (rect_class==4)*255)
            cv2.rectangle(resized, (x1, y1), (x2, y2), color, 1)
            cv2.putText(resized,box_dict['word'][index],(x1, y1),font,fontScale,color,thickness)
    
    cv2.imshow('Image',resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_grids(image_dir: str, box_dict: dict, nb_cols: int, nb_lines : int) -> dict:
    """Generates a grid of words per image, based on the boxes dictionary given as parameter.
    Returns the grid information in a dictionary with the format :
    { image_name : { (col,line) : [word, class] } }
    
    :param image_dir: images directory path
    :type image_dir: str
    :param box_dict: dictionary prepared with divide_boxes function
    :type box_dict: dict
    :param nb_cols: number of columns wanted in the grid
    :type nb_cols: int
    :param nb_lines: number of lines wanted in the grid
    :type nb_lines: int
    :return: dictionary containing the coordinates of the words in the grid 
    :rtype: dict

    """
    images = list(set(box_dict['filename']))
    grid = {}
    for image in images:
        grid[image] = {}
        
        indexes = [idx for (idx, name) in enumerate(box_dict['filename']) if name == image]
        x_values = [x for (idx, x) in enumerate(box_dict['x']) if idx in indexes]
        y_values = [x for (idx, x) in enumerate(box_dict['y']) if idx in indexes]

        xmin = min(x_values) - 1
        xmax = max(x_values) + 1
        ymin = min(y_values) - 1
        ymax = max(y_values) + 1

        im_height, im_width, _ = get_img_shape(os.path.join(image_dir,image))
        ticket_height = ymax - ymin
        ticket_width = xmax - xmin

        for idx in indexes:
            word_col = int(((box_dict['x'][idx] - xmin) / ticket_width) * nb_cols)
            word_line = int(((box_dict['y'][idx] - ymin) / ticket_height) * nb_lines)
            grid[image][(word_col,word_line)] = [box_dict['word'][idx], box_dict['class'][idx]]
    return grid

def visualize_grid(gridmap: dict, nb_cols: int, nb_lines : int, filename: str):
    """Opens a new window showing the visual representation of the image given as parameter.
    Image grid has to be prepared with the generate_grids function and given as parameter.
    
    :param gridmap: dictionary prepared with generate_grids function
    :type gridmap: dict
    :param nb_cols: number of columns in the grids
    :type nb_cols: int
    :param nb_lines: number of lines in the grids
    :type nb_lines: int
    :param filename: name of the image file to be checked
    :type filename: str

    """
    image = np.zeros((620,620,3), np.uint8) + 255
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    fontScale = 0.5
    thickness = 1
    for key, value in gridmap[filename].items():
        x = int((key[0] * 600) / nb_cols)
        y = int((key[1] * 600) / nb_lines) + 10
        cell_word = value[0]
        cell_class = value[1]
        color = ((cell_class==1)*50 + (cell_class==2)*255, (cell_class==1)*150 + (cell_class==3)*255\
                        ,(cell_class==1)*255 + (cell_class==4)*255)
        cv2.putText(image,cell_word,(x, y),font,fontScale,color,thickness)
    for line_idx in range(nb_lines):
        start_point = (0, int(line_idx*600/nb_lines) + 10)
        end_point = (620, int(line_idx*600/nb_lines) + 10)
        cv2.line(image, start_point, end_point, (0,0,0), 1)
    for col_idx in range(nb_cols):
        start_point = (int(col_idx*600/nb_cols + 30), 0)
        end_point = (int(col_idx*600/nb_cols) + 30, 620)
        cv2.line(image, start_point, end_point, (0,0,0), 1)

    cv2.imshow(filename, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# box_dict = divide_boxes('tools/cutie/1629378395_1_csv.csv')
# check_box_dictionary('tools/yolo/data/f94dd838-259e-4ac6-811c-92c61b0d80c4.jpg', box_dict)
# mygrid = generate_grids('tools/yolo/data',box_dict, 15, 30)
# files = [file for file in os.listdir('tools/yolo/data') if file[-1] == 'g']
# filename = '5d87e1bb-31fa-4e5e-baec-34dfbf937a99.jpg'
# visualize_grid(mygrid, 15, 30, files[1])

