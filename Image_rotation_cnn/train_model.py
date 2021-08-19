# Imports
import numpy as np
import pandas as pd
from tensorflow import keras 
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random
import os

# Constants
FAST_RUN = True
IMAGE_WIDTH = 180
IMAGE_HEIGHT = 180
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
IMAGE_FOLDER = 'Data'

#Datasets Preparation
filenames = os.listdir(IMAGE_FOLDER)
categories = []
for filename in filenames:
    category = filename.split('.')[0][-3:]
    if category == '000':
      categories.append('0deg')
    elif category == '090':
      categories.append('90deg')
    elif category == '180':
      categories.append('180deg')
    else:
      categories.append('270deg')

# DataFrame with all files and their class
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
    })

# Check the number of files of each class
df['category'].value_counts().plot.bar()
plt.show()

# Plot a sample image
sample = random.choice(filenames)
image = load_img(os.path.join(IMAGE_FOLDER,sample))
plt.imshow(image)
plt.show()

# Prepare the training, testing and validating datasets
train_df, other_df = train_test_split(df, test_size=0.3, random_state=42)
test_df, validate_df = train_test_split(other_df, test_size=0.4, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# Check the number of files of each class in each datasets
train_df['category'].value_counts().plot.bar()
plt.show()
test_df['category'].value_counts().plot.bar()
plt.show()
validate_df['category'].value_counts().plot.bar()
plt.show()

# Store some parameters
total_train = train_df.shape[0]
total_test = validate_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15


# Model initialization
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax')) # 2 because we have 4 classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]



# Training Generator
train_datagen = ImageDataGenerator(
    rotation_range=10,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    IMAGE_FOLDER, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# Testing Generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    IMAGE_FOLDER, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# Validating Generator
validate_datagen = ImageDataGenerator(rescale=1./255)
validate_generator = validate_datagen.flow_from_dataframe(
    validate_df, 
    IMAGE_FOLDER, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# See how the generator works
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    IMAGE_FOLDER, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# Fit model (uncomment below to launch the model training)
epochs=3 if FAST_RUN else 50
# history = model.fit_generator(
#     train_generator, 
#     epochs=epochs,
#     validation_data=test_generator,
#     validation_steps=total_test//batch_size,
#     steps_per_epoch=total_train//batch_size,
#     callbacks=callbacks
# )

# Saving model
model.save(os.path.join("model","model.keras"))


# Visualize training
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="Testing loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Testing accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

# Validation
validate_X = validate_df[['filename']]
nb_samples = validate_X.shape[0]
val_gen = ImageDataGenerator(rescale=1./255)
val_generator = val_gen.flow_from_dataframe(
    validate_X, 
    IMAGE_FOLDER, 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

predict = model.predict_generator(val_generator, steps=np.ceil(nb_samples/batch_size))
validate_X['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
validate_X['category'] = validate_X['category'].replace(label_map)

# Visualize some predictions
sample_val = validate_X.head(18)
plt.figure(figsize=(12, 24))
for index, row in sample_val.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(os.path.join(IMAGE_FOLDER,filename), target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename[-10:] + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()

# Print confusion matrix and classification report
print('Confusion Matrix')
cm = confusion_matrix(validate_df['category'], validate_X['category'])
print(cm)
print('Classification Report')
print(classification_report(validate_df['category'], validate_X['category']))
