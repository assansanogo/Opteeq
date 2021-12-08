import sys

from image_rotation.cnn.prepare_functions import prepare_training_images

if __name__ == '__main__':
    in_folder_parameter = sys.argv[1]
    out_folder_parameter = sys.argv[2]
    file_counter, not_worked = prepare_training_images(in_folder_parameter, out_folder_parameter)

    print(f'{file_counter} training images uploaded')

    if not_worked != []:
        print('The following files could not be processed :')
        print(str(not_worked))
