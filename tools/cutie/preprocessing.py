import json
import os
import typing

import cv2
import numpy as np
import pandas as pd
import torch

from tools.image.imageTools import get_img_shape
from tools.yolo.preprocessing import clean_via_file


def divide_boxes(via_file: str) -> dict:
    """
    Divides the boxes containing several words in the via annotations file given as parameter.
    Returns a dictionary with the information of each box : filename, class, word, x, y, width, height.

    :param via_file: path to the via csv file
    :type via_file: str
    :return: dictionary with the boxes information
    :rtype: dict
    """
    via_clean = clean_via_file(via_file)  # clean the initial file to get the right separators
    images_boxes = pd.read_csv(via_clean, sep='|', header=0)
    images_boxes = images_boxes[['filename', 'region_shape_attributes', 'region_attributes']]
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
        box_type = eval(images_boxes.at[index,'box_shape'])['name']
        if box_type == 'polygon':
            all_points_x = eval(images_boxes.at[index,'box_shape'])['all_points_x']
            all_points_y = eval(images_boxes.at[index,'box_shape'])['all_points_y']
            x_min = min(all_points_x)
            x_max = max(all_points_x)
            y_min = min(all_points_y)
            y_max = max(all_points_y)
            box_width = x_max - x_min
            box_height = y_max - y_min
        if box_type == 'rect':
            box_width = eval(images_boxes.at[index,'box_shape'])['width']
            box_height = eval(images_boxes.at[index,'box_shape'])['height']
            x_min = eval(images_boxes.at[index,'box_shape'])['x'] - (box_width / 2)
            x_max = x_min + box_width
            y_min = eval(images_boxes.at[index,'box_shape'])['y'] - (box_height / 2)
            y_max = y_min + box_height

        char_counter = 0
        if total_chars != 0:
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
    """
    Opens a new window showing the image and the boxes found in the dictionary given as parameter.

    :param image_path: path to the image
    :type image_path: str
    :param box_dict: dictionary prepared with divide_boxes function
    :type box_dict: dict
    """
    filename = os.path.basename(image_path)
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
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


def generate_grids(json_file: str, grid_size: int = 64, known_class: bool = True) -> dict:
    """
    Generates a grid of words per image, based on the json file containing the annotations.
    Returns the grid information in a dictionary with the format :
    { 'image' : str, 'grid' : { (col,line) : [word, class, dressed_word] } }

    :param json_file: path to the json file containing the annotations
    :type json_file: str
    :param grid_size: number of rows/columns in the output grid, default 64
    :type grid_size: int
    :param known_class: True if the classes are available in the json, False otherwise, default True
    :type known_class: bool
    :return: dictionary containing the image file name, the coordinates of the words in the grid and their class
    :rtype: dict

    """
    image_file_name = os.path.basename(json_file).split('.')[0] + '.jpg'
    final_dict = {}
    final_dict['image'] = image_file_name
    final_dict['grid'] = {}

    with open(json_file) as file:
      data = json.load(file)

    xmin_values = [box['bbox'][0] for box in data['text_boxes']]
    xmax_values = [box['bbox'][2] for box in data['text_boxes']]
    ymin_values = [box['bbox'][1] for box in data['text_boxes']]
    ymax_values = [box['bbox'][3] for box in data['text_boxes']]
    xmin = min(xmin_values) - 1
    xmax = max(xmax_values) + 1
    ymin = min(ymin_values) - 1
    ymax = max(ymax_values) + 1
    ticket_height = ymax - ymin
    ticket_width = xmax - xmin

    for box in data['text_boxes']:
      x = (box['bbox'][0] + box['bbox'][2]) / 2
      y = (box['bbox'][1] + box['bbox'][3]) / 2
      box_id = box['id']
      if known_class:
        box_class = 'NOTHING'
        for field in data['fields']:
          if box_id in field['value_id']:
            box_class = field['field_name']
            break
      else:
        box_class = '?'

      word_col = int(((x - xmin) / ticket_width) * grid_size)
      word_line = int(((y - ymin) / ticket_height) * grid_size)
      check_existing_key = False
      while check_existing_key == False:
        if (word_col,word_line) not in final_dict['grid']:
          final_dict['grid'][(word_col,word_line)] = [box['text'], box_class, box['dressed_text']]
          check_existing_key = True
        else:
           word_col += 1

    return final_dict


def visualize_grid(gridmap: dict, nb_cols: int, nb_lines: int):
    """
    Opens a new window showing the visual representation of the image given as parameter.
    Image grid has to be prepared with the generate_grids function and given as parameter.

    :param gridmap: dictionary prepared with generate_grids function
    :type gridmap: dict
    :param nb_cols: number of columns in the grids
    :type nb_cols: int
    :param nb_lines: number of lines in the grids
    :type nb_lines: int

    """
    image = np.zeros((620,620,3), np.uint8) + 255
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    fontScale = 0.5
    thickness = 1
    for key, value in gridmap['grid'].items():
        x = int((key[0] * 600) / nb_cols)
        y = int((key[1] * 600) / nb_lines) + 10
        cell_word = value[0]
        cell_class = value[1]
        color = {'NOTHING': (0,0,0), 'PLACE': (255,0,0), 'DATE': (0,255,0), 'TOTAL_TEXT': (0,255,255), 'TOTAL_AMOUNT': (0,0,255), '?': (0,0,0)}
        cv2.putText(image,cell_word,(x, y),font,fontScale,color[cell_class],thickness)
    for line_idx in range(nb_lines):
        start_point = (0, int(line_idx*600/nb_lines) + 10)
        end_point = (620, int(line_idx*600/nb_lines) + 10)
        cv2.line(image, start_point, end_point, (0,0,0), 1)
    for col_idx in range(nb_cols):
        start_point = (int(col_idx*600/nb_cols + 30), 0)
        end_point = (int(col_idx*600/nb_cols) + 30, 620)
        cv2.line(image, start_point, end_point, (0,0,0), 1)

    cv2.imshow('grid',image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def generate_cutie_jsons(via_file: str):
    """
    Generate the json files of the images contained in an annotation file,
    in the needed format for cutie training.

    :param via_file: path to the via csv file
    :type via_file: str

    """
    box_dict = divide_boxes(via_file)
    images = list(set(box_dict['filename']))
    class_map = {1: 'PLACE', 2: 'TOTAL_TEXT', 3: 'TOTAL_AMOUNT', 4: 'DATE'}

    for image in images:
        field_init = [{'field_name': field, 'value_id':[], 'value_text':[], 'key_id': [], 'key_text' : []} \
            for field in class_map.values()]
        result_dict = {'text_boxes' : [], 'fields' : field_init,'global_attributes' : {'file_id' : image}}
        indexes = [idx for (idx, name) in enumerate(box_dict['filename']) if name == image]
        cutie_idx = 0
        for idx in indexes:
            cutie_idx += 1
            cutie_xmin = box_dict['x'][idx] - (box_dict['width'][idx] / 2)
            cutie_xmax = box_dict['x'][idx] + (box_dict['width'][idx] / 2)
            cutie_ymin = box_dict['y'][idx] - (box_dict['height'][idx] / 2)
            cutie_ymax = box_dict['y'][idx] + (box_dict['height'][idx] / 2)
            cutie_word = box_dict['word'][idx]
            cutie_class = box_dict['class'][idx]

            text_box = {'id' : cutie_idx, 'bbox' : [cutie_xmin , cutie_ymin, cutie_xmax, cutie_ymax], 'text' : cutie_word }
            result_dict['text_boxes'].append(text_box)
            if cutie_class in class_map.keys():
                result_dict['fields'][cutie_class - 1]['value_id'].append(cutie_idx)
                result_dict['fields'][cutie_class - 1]['value_text'].append(cutie_word)

        with open(image.split('.')[0] + '.json', 'w') as outfile:
            json.dump(result_dict, outfile)
        result_dict.clear()


def generate_1_vocab(json_file: str, grid_size: int):
    """
    Generates the vocabulary of the words contained in a json annotation file.

    :param json_file: path to the json_file file
    :type json_file: str
    :param grid_size: grid size that will be used in Cutie
    :type grid_size: int
    :return: list of the sentences found in the json file (1 sentence / line)
    :rtype: list
    """
    grid = generate_grids(json_file, grid_size, known_class=False)
    vocab = []
    for line in range(1,grid_size+1):
      sentence = []
      for col in range(1,grid_size+1):
        if (col,line) in grid['grid'].keys():
          sentence.append(grid['grid'][(col,line)][2])
      if len(sentence) != 0:
        vocab.append(sentence)
    return vocab


def generate_vocab(folder: str, grid_size: int):
    """
    Generates the vocabulary of the words contained in all the json annotation files from a folder.

    :param folder: path to the folder
    :type folder: str
    :param grid_size: grid size that will be used in Cutie
    :type grid_size: int
    :return: list of the sentences found in the json files (1 sentence / line)
    :rtype: list
    """
    files = [file for file in os.listdir(folder) if file.endswith('.json')]
    vocab = []
    for file in files:
        current_vocab = generate_1_vocab(os.path.join(folder,file), grid_size)
        vocab.extend(current_vocab)
    return vocab


def is_number(s: str):
    """
    Asserts if the string passed as parameter is a number

    :param s: string to be checked
    :type s: str
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def dress_text(text: str):
    """
    Dress a text, replacing numbers by '0'

    :param text: string to be dressed
    :type text: str
    :return: dressed string
    :rtype: str
    """
    string = text.lower()
    for idx, letter in enumerate(string):
        if is_number(letter):
            string = string[:idx] + '0' + string[idx + 1:]
    return string


def generate_cutie_jsons(via_file: str):
    """
    Generate the json files of the images contained in an annotation file,
    in the needed format for cutie training.

    :param via_file: path to the via csv file
    :type via_file: str
    :param embedding_fun: word embedding function
    :type embedding_fun: callable

    """
    box_dict = divide_boxes(via_file)
    images = list(set(box_dict['filename']))
    class_map = {1: 'PLACE', 2: 'TOTAL_TEXT', 3: 'TOTAL_AMOUNT', 4: 'DATE'}

    for image in images:
        field_init = [{'field_name': field, 'value_id':[], 'value_text':[], 'key_id': [], 'key_text' : []} \
            for field in class_map.values()]
        result_dict = {'text_boxes' : [], 'fields' : field_init,'global_attributes' : {'file_id' : image}}
        indexes = [idx for (idx, name) in enumerate(box_dict['filename']) if name == image]
        cutie_idx = 0
        for idx in indexes:
            cutie_idx += 1
            cutie_xmin = box_dict['x'][idx] - (box_dict['width'][idx] / 2)
            cutie_xmax = box_dict['x'][idx] + (box_dict['width'][idx] / 2)
            cutie_ymin = box_dict['y'][idx] - (box_dict['height'][idx] / 2)
            cutie_ymax = box_dict['y'][idx] + (box_dict['height'][idx] / 2)
            cutie_word = box_dict['word'][idx]
            cutie_dressed_word = dress_text(cutie_word)
            cutie_class = box_dict['class'][idx]
            text_box = {'id' : cutie_idx, 'bbox' : [cutie_xmin , cutie_ymin, cutie_xmax, cutie_ymax], 'text' : cutie_word, 'dressed_text': cutie_dressed_word }
            result_dict['text_boxes'].append(text_box)
            if cutie_class in class_map.keys():
                result_dict['fields'][cutie_class - 1]['value_id'].append(cutie_idx)
                result_dict['fields'][cutie_class - 1]['value_text'].append(cutie_word)

        with open(image.split('.')[0] + '.json', 'w') as outfile:
            json.dump(result_dict, outfile)
        result_dict.clear()

def distilbert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    vec = model(**inputs).last_hidden_state[0,0]
    return vec


def convert_json_to_tensors(json_file: str, embedding_fun: typing.Callable, grid_size: int = 64,
                            embedding_size: int = 128, N_class: int = 5):
    """
    Converts the annotations json file to two tensors : one for the grid with the word embeddings, one
    for the classes of the words.

    :param json_file: dictionary prepared with generate_grids function
    :type json_file: dict
    :param embedding_fun: function converting a word to its embedding vector
    :type embedding_fun: function
    :param grid_size: number of rows/columns in the output grid, default 64
    :type grid_size: int
    :param embedding_size: size of the word embedding vectors, default = 128
    :type embedding_size: int
    :param N_class: Number of classes to predict, default = 5
    :type N_class: int
    :return: The two tensors of the grid and the classes
    :rtype: tuple(torch.tensor torch.tensor)

    """
    grid = generate_grids(json_file, grid_size)
    with torch.no_grad():
        grid_tensor = torch.zeros([grid_size, grid_size, embedding_size])
        classes_tensor = torch.zeros([N_class, grid_size * grid_size],dtype=torch.float)

        class_map = {'NOTHING' : 0, 'PLACE': 1, 'DATE': 2, 'TOTAL_TEXT': 3, 'TOTAL_AMOUNT': 4}
        for key in grid['grid'].keys():
            x_idx = key[0] - 1
            y_idx = key[1] - 1
            word_vector = embedding_fun(grid['grid'][key][0])
            word_class = class_map[grid['grid'][key][1]]
            grid_tensor[x_idx, y_idx] = word_vector

            classes_tensor[word_class, y_idx * grid_size + x_idx] = 1
        grid_tensor = torch.permute(grid_tensor, (2, 0, 1))

    return grid_tensor, classes_tensor

