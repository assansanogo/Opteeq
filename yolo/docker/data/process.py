import glob
import os

# Current directory
current_dir = "obj"

# Percentage of images to be used for the test set
percentage_test = 0.2

# Create and/or truncate train.txt and test.txt
file_train = open("train.txt", "w")
file_test = open("test.txt", "w")
file_validation = open("validation.txt", "w")

counter = 1

img = glob.glob(os.path.join(current_dir, "*.jpg"))
img_unique = set([i.replace(".jpg", "").split("_")[0] for i in img])
total = len(img_unique)
for i in img_unique:
    if counter > total * (1 - percentage_test):
        counter += 1
        file_test.write(i + ".jpg" + "\n")
    elif counter > total * (1 - 2 * percentage_test):
        counter += 1
        file_validation.write(i + ".jpg" + "\n")
    else:
        file_train.write(i + ".jpg" + "\n")
        file_train.write(i + "_b" + ".jpg" + "\n")
        file_train.write(i + "_n" + ".jpg" + "\n")
        counter += 1
