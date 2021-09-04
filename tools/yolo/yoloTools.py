from tools.image.imageTools import get_img_shape
import csv

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

def convert_via_to_yolo(via_file: str) -> str:
    """Converts annotation csv file from via to a txt file with yolo input format. 
    Returns the path to the txt file .

    :param via_file: path to the via csv file
    :type via_file: str
    :return: path to the txt file formatted for yolo
    :rtype: str
    """
    via_clean = clean_via_file(via_file)
    images = get_image_names(via_clean)
    for image in images:
        '''TO DO
        1/ Download image from s3
        2/ Get the size of the image
        3/ Calculate coordinates of each box (x,y of the box center + width, height). 
        Coordinates must be relative (from 0.0 to 1.0)
        4/ Create a txt with same name as the image and one line / box with this
        format : <object-class> <x> <y> <width> <height>
        ex : 1 0.716797 0.395833 0.216406 0.147222
             0 0.687109 0.379167 0.255469 0.158333
             1 0.420312 0.395833 0.140625 0.166667
        '''


'''
via_file = 'tools/yolo/1629378395_1_csv.csv'
file_path = clean_via_file(via_file)
import pandas as pd
dat = pd.read_csv('tools/yolo/1629378395_1_csv_clean.csv',sep='|',header=0)
dat.columns
d = eval(dat.region_shape_attributes[0].strip('"').replace('""""','"'))
(max(d['all_points_x'])-min(d['all_points_x']))/2
dat.head()
'''