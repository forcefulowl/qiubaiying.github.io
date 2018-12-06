---
layout: post
title:  Skin Lesion Analysis Towards Melanoma Detection
subtitle:   optimized for mobile devices
date:   2018-12-03
author: gavin
header-img: img/shufflenet.jpg
catalog:    true
tags:
    - deep learning
---
.

# Background

This topic is a challenge annocuned by ISIC. The goal of this recurring challenge is to help participants develop image analysis tools to enable the automated diagnosis of melanoma from dermoscopic images.

### About the ISIC archive

The International Skin Imaging Collaboration (ISIC) is an international effort to improve melanoma diagnosis, sponsored by the International Society for Digital Imaging of the Skin (ISDIS). The ISIC Archive contains the largest publicly available collection of quality controlled dermoscopic images of skin lesions.

Presently, the ISIC Archive contains over 13,000 dermoscopic images, which were collected from leading clinical centers internationally and acquired from a variety of devices within each center. Broad and international participation in image contribution is designed to insure a representative clinically relevant sample.

All incoming images to the ISIC Archive are screened for both privacy and quality assurance. Most images have associated clinical metadata, which has been vetted by recognized melanoma experts. A subset of the images have undergone annotation and markup by recognized skin cancer experts. These markups include dermoscopic features (i.e., global and focal morphologic elements in the image known to discriminate between types of skin lesions).

### About Melanoma

Skin cancer is a major public health problem, with over 5,000,000 newly diagnosed cases in the United States every year. Melanoma is the deadliest form of skin cancer, responsible for an overwhelming majority of skin cancer deaths. In 2015, the global incidence of melanoma was estimated to be over 350,000 cases, with almost 60,000 deaths. Although the mortality is significant, when detected early, melanoma survival exceeds 95%.

# Data

The input data are dermoscopic lesion images in JPEG format.

The training data consists of 10015 images.

The format of raw data is as follows:

<img src='/img/mobilenet/1.raw_data.png'>

And the format of the label is as follows:

<img src='/img/mobilenet/2.raw_label.png'>

But if I directly load all of the data into memory that is so memory consuming, even the most state-of-the art configuration won't have enough memory space to process the data the way I used to do it. Meanwhile, the number of training data is not large enough, so I also wanna do Data Augumentation.

Firstly, I change the format of the raw data.

<img src='/img/mobilenet/3.new_data.png'>

The code to do that is as follows:

```
f = open('C:\\Users\gavin\Desktop\labels.txt')
line = f.readline()
labels = []

while line:
    if line[13] == '1':
        labels.append(0)
    elif line[15] == '1':
        labels.append(1)
    elif line[17] == '1':
        labels.append(2)
    elif line[19] == '1':
        labels.append(3)
    elif line[21] == '1':
        labels.append(4)
    elif line[23] == '1':
        labels.append(5)
    elif line[25] == '1':
        labels.append(6)
    line = f.readline()

path = 'C:\\Users\gavin\Desktop\ISIC2018_Task3_Training_Input'
count = 0
while count < len(labels):
    curr_num = str(24306+count)
    if labels[count] == 0:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\MEL')
    if labels[count] == 1:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), r'C:\\Users\gavin\Desktop\Train\NV')
    if labels[count] == 2:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\BCC')
    if labels[count] == 3:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\AKIEC')
    if labels[count] == 4:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\BKL')
    if labels[count] == 5:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\DF')
    if labels[count] == 6:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\VASC')
    count = count + 1
```

Then I do the Data Augumentation:

```
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)


test_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)
```

Then achieve ImageGenerator:

```
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical")
```

# Model

For my case, I use Transfer Learning. Transfer learning, is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. I use transfer learning because it's rare to get enough dataset, so, using pre-trained network weights as initialisations or a fixed feature extractor helps in solving problems.


### Structure of Model

```
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation

img_width, img_height = 224, 224
train_data_dir = "C:\\Users\gavin\Desktop\Train1_M"
validation_data_dir = "C:\\Users\gavin\Desktop\Test1_M"
data_dir = 'C:\\Users\gavin\Desktop\whole_M'
nb_train_samples = 7000
nb_validation_samples = 3000
batch_size = 32
epochs = 100

model = applications.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation="softmax")(x)

# creating the final model
model_final = Model(input=model.input, output=predictions)
```
I use 'imagenet' weights as initial weights, 'include_top=False' means remove the fully-connected layer at the top of the network. Because in my case, the total classes is 7.

```
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])

model_final.fit_generator(
    train_generator,
    steps_per_epoch=6900//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=3000//batch_size)
```
Because I use `ImageDataGenerator` before, so here I have to use `fit_generator` to train the model.

The result of the model is as follows:

<img src>

According to the result, we may suffer overfitting.

### Overfitting

Firstly I changed 'batch_size', 'learning rate' and 'epoch', but it did not work.

##### Change validation dataset

Because I split training and validation dataset manually, I just choose the last 30% data as validation data.

<img src='/img/mobilenet/4.split_data.png'>

Then I tried to choose the first 30% data as validation data. 

Except above, I thought maybe I should not split them manually, then I changed the ImageDataGenerator so the system can split it auto.

```
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30,
    validation_split=0.3)
    
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical"
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation')
```

'validation_split=0.3' means split data to 70% for training and 30% for validation.

'subset='training', 'subset='validation' to determine which dataset it use.

##### BatchNormalization

I added 'BatchNormalization' after 'Dense' and before 'Activation'.

```
model = applications.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

x = model.output
x = Flatten()(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation="softmax")(x)
```

##### L1, L2 Regularization

I also tried l1, l2 Regularization in 'Dense'

```
from keras import regularizers

model = applications.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

x = model.output
x = Flatten()(x)
x = Dense(1024, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation="softmax")(x)
```

##### Early Stop

```
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model_final.fit_generator(
    train_generator,
    steps_per_epoch=7000//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=3000//batch_size,
    callbacks=[early_stop])
```

<img src='/img/mobilenet/early_stop.png'>

##### Change optimizer

```
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0), metrics=['accuracy'])
```












