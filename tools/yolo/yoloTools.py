def convert_via_to_yolo(via_file: str) -> str:
    """Converts annotation csv file from via to a txt file with yolo input format. 
    Returns the path to the txt file .

    :param via_file: path to the via csv file
    :type via_file: str
    :return: path to the txt file formatted for yolo
    :rtype: str
    """
    file_path = clean_via_file(via_file)
    #TO DO : Implement the function

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
    return(out_file)

via_file = 'tools/yolo/1629378395_1_csv.csv'
file_path = clean_via_file(via_file)

'''
import pandas as pd
dat = pd.read_csv('tools/yolo/1629378395_1_csv_clean.csv',sep='|',header=0)
dat.columns
d = eval(dat.region_shape_attributes[0].strip('"').replace('""""','"'))
(max(d['all_points_x'])-min(d['all_points_x']))/2
dat.head()
'''