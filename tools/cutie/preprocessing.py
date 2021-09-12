from tools.yolo.preprocessing import clean_via_file
from tools.image.imageTools import get_img_shape
import cv2
import pandas as pd
import os

def divide_boxes(via_file: str) -> dict:
    """Divides the boxes containing several words in the via annotations file given a s parameter.
    Returns a dictionnary with the information of each box : filename, class, word, x, y, width, height.

    :param via_file: path to the via csv file
    :type via_file: str
    :return: dictionnary with the boxes information
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

def check_box_dictionnary(image_path: str, box_dict: dict):
    """Opens a new window showing the image and the boxes found in the dictionnary given as parameter.
    
    :param image_path: path to the image
    :type image_path: str
    :param box_dict: dictionnary prepared with divide_boxes function
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

mydict = divide_boxes('tools/cutie/1629378395_1_csv.csv')
check_box_dictionnary('tools/yolo/data/f94dd838-259e-4ac6-811c-92c61b0d80c4.jpg', mydict)