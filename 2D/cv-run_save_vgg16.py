import argparse

parser = argparse.ArgumentParser(description="run",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--image_len", default=32, type=int, help="IMAGE_LEN")
parser.add_argument("-b", "--batch_size", default=32, type=int,  help="BATCH_SIZE")
parser.add_argument("-t", "--folder_train",default='train',  help="folder_train")
parser.add_argument("-r", "--folder_test", default='test',  help="folder test")
parser.add_argument("-e", "--num_epoch", default=3, type=int, help="epoch")
parser.add_argument("-p", "--patience_epoch", default=3, type=int, help="epoch patience")
parser.add_argument("-c", "--num_class", default=2, type=int, help="number of classes")
parser.add_argument("-l", "--learning_rate", default=0.0001, type=float, help="number of classes")
parser.add_argument("-k", "--nkfold", default=0, type=int, help="kfold ")
parser.add_argument("-f", "--folder", default='save_models', help="folder to save")
parser.add_argument("--seed", default=1,type=int, help="folder to save")
parser.add_argument("--mod", default='vgg', help="model")
parser.add_argument("--nfilter", default=64, type=int, help="number of filter (for cnn3)")

args = parser.parse_args()
config = vars(args)
print(config)


IMAGE_LEN = args.image_len
BATCH_SIZE = args.batch_size
folder_train = args.folder_train
folder_test = args.folder_test
path_2 = folder_train + folder_test
path_named = path_2.replace("/", ".")
num_epoch = args.num_epoch
num_class = args.num_class
learning_rate=args.learning_rate
patience_epoch=args.patience_epoch
nkfold=args.nkfold
folder_save = args.folder
seed_v = args.seed
mod = args.mod


import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
#khai báo frame work tensorflow
import tensorflow as tf
import random
#import keras từ frame work tensorflow
from tensorflow import keras
import tensorflow.keras 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, MaxPool2D
from tensorflow.keras.layers import Conv2D, InputLayer
from tensorflow.keras.layers import MaxPooling2D


# gan cac seed de co kq thuc nghiem giong

tf.random.set_seed(seed_v)
random.seed(seed_v)
np.random.seed(seed_v)

import time
start = time.time()


## tim tap tin de tranh bi trung
import os, fnmatch
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result



#512x512 size của ảnh

IMAGE_SIZE = (IMAGE_LEN, IMAGE_LEN)
#chia dữ liệu huấn luyện/ kiểm thử thành từng batch

#tiền sử lý ảnh
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #link dẫn ảnh
    folder_train,
    #chia chia train và val
    #validation_split=0,
    #subset="training",
    label_mode = "categorical",
    #seed=1,
    #size của ảnh
    image_size=IMAGE_SIZE,
    #batch_size : chỉa ảnh vào từng batch để trainning như trong bài là 32
    batch_size=BATCH_SIZE)
#tiền sử lý ảnh
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #link dẫn ảnh
    folder_test,
    #chia chia train và val
    #validation_split=0,
    #subset="training",
    label_mode = "categorical",
    #seed=1,
    #size của ảnh
    image_size=IMAGE_SIZE,
    #batch_size : chỉa ảnh vào từng batch để trainning như trong bài là 32
    batch_size=BATCH_SIZE)


from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
#from keras.utils import to_categorical
from sklearn import preprocessing
#from exploit_pred import *



def model_effi(num_classes = num_class,   image_size = IMAGE_LEN, batch_size = BATCH_SIZE):


    img_input = Input(shape=(image_size,image_size,3))

    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='dpacontest_v4')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


if nkfold in [0,1]:

    if mod == 'vgg':
        name_model = ''
    else:
        name_model = mod + '_'
    name_saved= name_model + path_named + '_c'+ str(num_class)+ '_s' + str(IMAGE_LEN) + '_b'+ str(BATCH_SIZE) +  '_e'+ str(num_epoch) +'_p'+ str(patience_epoch) +  '_lr'+str(learning_rate) + '_se'+ str(seed_v) +'_k'+ str(nkfold) + 'nfilter'+str(args.nfilter) 
    print ('name_saved='+name_saved)
    n_files = find(name_saved + '*.json',folder_save )
    if len(n_files)>0:
        #print('name_saved'+name_saved) #dung thuc nghiem neu lam roi
        print('thuc nghiem '+name_saved+' da lam roi10!')
        exit()

    if mod == 'fc':
            
        # Build the model
        model = Sequential()       
        model.add(InputLayer(input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))  
        model.add(Flatten())
        model.add(Dense(units=num_class, activation="softmax"))

        # Compile the model
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        
        from tensorflow.keras.callbacks import EarlyStopping
        # Define Early Stopping callback
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        print('fc')
        model.summary()

    elif mod =='cnn001_k':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
       

        # Print model summary
        print('cnn001_k')
        model.summary()
        
        
    elif mod =='cnn001_m':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
            
            
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, classification_report
        # Predict the classes for the test set
        y_pred = model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred_classes)
        # Plot confusion matrix as a heatmap
        plt.figure(figsize=(12.5,10.5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=train_generator.class_indices.keys(), yticklabels=train_generator.class_indices.keys())
  
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.show()
        
        # Print model summary
        print('cnn001_m')
        model.summary()
        
        
        
        
    elif mod =='cnn001':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Build the model
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=NUM_CLASSES, activation='softmax'))

        # Compile the model
        opt = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        # Train the model
        history = model.fit(
            train_generator,
            epochs=NUM_EPOCHS,
            validation_data=test_generator,
            callbacks=[early_stop, reduce_lr]
        )

        # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        # Print model summary
        print('cnn001')
        model.summary()
        
        
        
    elif mod =='cnn002':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Build the model 
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=NUM_CLASSES, activation='softmax'))
        

        # Compile the model
        opt = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        # Train the model
        history = model.fit(
            train_generator,
            epochs=NUM_EPOCHS,
            validation_data=test_generator,
            callbacks=[early_stop, reduce_lr]
        )

        # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        # Print model summary
        print('cnn002')
        model.summary()
        
    elif mod =='cnn002_k':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

        # Print model summary
        print('cnn002_k')
        model.summary()
    
    
    elif mod =='cnn002_m':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

        # Print model summary
        print('cnn002_m')
        model.summary()
        
    
    
    elif mod =='cnn003':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Build the model
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=NUM_CLASSES, activation='softmax'))

        # Compile the model
        opt = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        # Train the model
        history = model.fit(
            train_generator,
            epochs=NUM_EPOCHS,
            validation_data=test_generator,
            callbacks=[early_stop, reduce_lr]
        )

        # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        # Print model summary
        print('cnn003')
        model.summary()
        
    
    
    elif mod =='cnn003_k':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))            
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
            model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
       

        # Print model summary
        print('cnn003_k')
        model.summary()
        
        
    elif mod =='cnn003_m':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))    
            model.add(MaxPool2D(pool_size=(2, 2)))        
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
       

        # Print model summary
        print('cnn003_m')
        model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epoch)   
    history = model.fit(train_ds, validation_data=test_ds, epochs=num_epoch, verbose=1,callbacks=[callback])
    
    from datetime import datetime
    now=datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    #print("date and time:",date_time)

    # name with hyperparameters used
    name_saved=  name_saved +'_' + str(date_time) #+'_k'+str(i)


    print('Model Saved!')

    model.save(folder_save+'/'+ name_saved +'.h5')
    model_json = model.to_json()
    with open( folder_save + '/' + name_saved +".json", "w") as json_file:
        json_file.write(model_json)

        # luu lai log
    end = time.time()
    print("time run: ", end - start)

    ep_arr=range(1, len(history.history['accuracy'])+1, 1)
    idx = len(history.history['accuracy'])-1 #index of mang
    train_acc = history.history['accuracy']
    val_acc= history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']


    title_cols = np.array(["ep","train_acc","valid_acc","train_loss","valid_loss"])      

    res = (ep_arr, train_acc, val_acc, train_loss, val_loss)
    res = np.transpose(res)
    combined_res = np.array(np.vstack((title_cols, res)))
    
    log_name1 = name_saved +'s1'
    np.savetxt(folder_save + '/'+log_name1 +".txt", combined_res, fmt="%s",delimiter="\t") 
    
    #log 2 luu lai tham so va cac thong tin ve sample
    log_name2 = name_saved +'s2_'  + 'time'+ str(round(end - start,2)) + 'acc' +str(round(val_acc[idx],3))
    #np.savetxt('save_models/'+log_name+"log2.txt", args, fmt="%s",delimiter="\t")
    with open(folder_save + '/'+log_name2+ ".txt", 'w') as f:
        f.write(str(args))
    title_cols = np.array(["samples_train","samples_test","train_acc","train_loss","val_acc","val_loss"])  
    
    
    train_labels = np.concatenate(list(train_ds.map(lambda x, y:y)))
    test_labels = np.concatenate(list(test_ds.map(lambda x, y:y)))
    
    res=(len(train_labels),len(test_labels),train_acc[idx],train_loss[idx],val_acc[idx],val_loss[idx])
    res=np.transpose(res)
    combined_res=np.array(np.vstack((title_cols,res)))

    with open(folder_save+ '/'+log_name2+ ".txt", "ab") as f:
        f.write(b"\n")
        np.savetxt(f, combined_res, fmt="%s",delimiter="\t")     