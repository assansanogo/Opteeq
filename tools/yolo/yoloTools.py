from tools.image.imageTools import get_img_shape
from tools.aws.awsTools import Bucket
import pandas as pd
import csv
import json
import os

def clean_via_file(via_file: str) -> str:
    """Cleans annotation csv file from via to differentiate the separators "," from the actual ","
    Returns the path to the clean csv file .

    :param via_file: path to the via csv file
    :type via_file: str
    :return: path to the clean csv file
    :rtype: str
    """
    out_file = via_file[:-4]+'_clean.csv'
    with open(via_file[:-4]+'_clean.csv', 'w') as cleanfile:
        with open(via_file, 'r') as file:
            lines = file.readlines()
            _ = cleanfile.writelines(lines[0].replace(',','|'))
            for line in lines[1:]:
                curly_brackets = 0
                for character in line[1:-2]:
                    if character == '{':
                        curly_brackets += 1
                        _ = cleanfile.write(character)
                    elif character == '}':
                        curly_brackets -= 1
                        _ = cleanfile.write(character)
                    else:
                        if curly_brackets == 0:
                            _ = cleanfile.write(character.replace(',','|'))
                        else:
                            _ = cleanfile.write(character)
                _ = cleanfile.write("\n")
    return out_file

def get_image_names(clean_via_file: str) -> 'list[str]':
    """Returns a list of the image file names annotated in the cleaned via file given as input.
    
    :param clean_via_file: path to the via csv file cleaned with the clean_via_file function
    :type clean_via_file: str
    :return: list of the image file names
    :rtype: list[str]
    """
    images_list = set()
    with open(clean_via_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            if row[0].endswith('.jpg'):
                images_list.add(row[0])
    images_list = list(images_list)
    return images_list

def cut_overlength(center: float, length: float) -> 'tuple[float,float]':
    """Trims a box in case it goes outside of the image and returns
    the new coordinates of the center and length.
    
    :param center: coordinate (x or y) of the box center
    :type center: float
    :param length: width or height of the box
    :type length: float
    :return: new values of the center and of the length
    :rtype: tuple[float, float]
    """
    if (center - (length/2) < 0) and (center + (length/2) <= 1) :
        overlength = (length/2) - center
        new_center = center + overlength/2
        new_length = length - overlength
    elif (center - (length/2) >= 0) and (center + (length/2) > 1) :
        overlength = center + (length/2) - 1
        new_center = center - overlength/2
        new_length = length - overlength
    elif (center - (length/2) < 0) and (center + (length/2) > 1) :
        new_center = 0.5
        new_length = 1
    else:
        new_center = center
        new_length = length
    return (new_center,new_length)

def convert_via_to_yolo(via_file: str, out_dir: str):
    """Downloads the images annotated in the via csv file and creates one txt file / image
    with the class and coordinates of the boxes as per yolo input format.

    :param via_file: path to the via csv file
    :type via_file: str
    :param out_dir: path to the outputs directory
    :type out_dir: str
    """
    via_clean = clean_via_file(via_file)
    images = get_image_names(via_clean)
    
    with(open("conf.json", "r")) as f:
        conf = json.load(f)
    if conf["bucket_standardized"] and conf['profile']:
        images_bucket = Bucket(conf["bucket_standardized"], conf['profile'])
    else:
        print("edit config file and add missing arguments")
    
    images_boxes = pd.read_csv(via_clean,sep='|',header=0)
    images_boxes = images_boxes[['filename','region_shape_attributes','region_attributes']]

    class_mapping = {'6': 0, '1': 1, '2': 0, '3': 3, '4':2}

    for image in images:
        # Download image from s3 :
        images_bucket.download(image,out_dir)
        # Get image size :
        height, width, _ = get_img_shape(os.path.join(out_dir,image))
        # Filter DataFrame to get the rows of this image :
        boxes = images_boxes[images_boxes['filename'] == image]
        # Prepare path of the txt file :
        txt_file = os.path.join(out_dir,image[:-3]+'txt')

        for index, box in boxes.iterrows():
            box_shape = eval(box.region_shape_attributes.strip('"').replace('""""','"'))
            box_attributes = eval(box.region_attributes.strip('"').replace('""""','"'))
            box_class = class_mapping[box_attributes['type']]
            x = round((max(box_shape['all_points_x']) + min(box_shape['all_points_x'])) / (2 * width),6)
            y = round((max(box_shape['all_points_y']) + min(box_shape['all_points_y'])) / (2 * height),6)
            box_width = round((max(box_shape['all_points_x']) - min(box_shape['all_points_x'])) / (width),6)
            box_height = round((max(box_shape['all_points_y']) - min(box_shape['all_points_y'])) / (height),6)
            
            x, box_width = cut_overlength(x, box_width)
            y, box_height = cut_overlength(y, box_height)
            with open(txt_file, 'a') as txt:
                _ = txt.write(f'{box_class} {x} {y} {box_width} {box_height}\n')


        '''TO DO

        3/ Calculate coordinates of each box (x,y of the box center + width, height). 
        Coordinates must be relative (from 0.0 to 1.0)
        4/ Create a txt with same name as the image and one line / box with this
        format : <object-class> <x> <y> <width> <height>
        ex : 1 0.716797 0.395833 0.216406 0.147222
             0 0.687109 0.379167 0.255469 0.158333
             1 0.420312 0.395833 0.140625 0.166667
        '''



via_file = 'tools/yolo/1629378395_1_csv.csv'
out_dir = 'tools/yolo/data'
convert_via_to_yolo(via_file, out_dir)


'''