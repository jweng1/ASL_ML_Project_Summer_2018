from model import lrcnModel
import numpy as np
import os
from skimage.io import imread
from keras.utils import to_categorical
from KerasBatchGenerator import KerasBatchGenerator


def load_data(folder_path, dictionary):
    if 'train' in folder_path:
        names = ['Dana', 'liz', 'Jamiee', 'Naomi', 'Tyler','Lana']
    elif 'test' in folder_path:
        names = ['Dana', 'liz', 'Jamiee', 'Naomi', 'Tyler','Lana']
    else:
        raise ValueError('folder_path must be for train or test only')
    folders = os.listdir(folder_path)
    X, y = [], []
    for folder in folders:
        if folder[0] == '.':
            continue
        files = os.listdir(os.path.join(folder_path, folder))
        for file in files:
            if file.endswith('.jpg'):
                basename, ext = file.split('.')
                for name in names:
                    if name in basename:
                        img_path = os.path.join(folder_path, folder, file)
                        img = imread(img_path)
                        X.append(img)
                        y.append(dictionary[folder])
    return np.asarray(X), np.asarray(y)


def make_dictionary():
    dictionary = {}
    with open('sign_list') as f:
        for line in f.readlines():
            if '.' in line:
                idx, word = line.split('.')
                word = word.strip()
                dictionary[word] = int(idx) - 1
    return dictionary



def train(seq_length, model, saved_model=None, class_limit=None, image_shape=None,
    batch_size=32, nb_epochs=100):

     # dictionary = make_dictionary()
     # X_train, y_train = load_data('train', dictionary)
     # X_test, y_test = load_data('test', dictionary)
     # np.save('X_train.npy', X_train)
     # np.save('y_train.npy', y_train)
     # np.save('X_test.npy', X_test)
     # np.save('y_test.npy', y_test)

    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    y_train.shape #19385
    y_test.shape#4620

    y_train_onehot = to_categorical(y_train)
    y_test_onehot = to_categorical(y_test)

    train_data_generator = KerasBatchGenerator(X_train, y_train_onehot, batch_size)
    test_data_generator = KerasBatchGenerator(X_test, y_test_onehot, batch_size)

    lrcnm = lrcnModel(59, model, seq_length, saved_model)

    lrcnm.model.fit_generator(train_data_generator.frame_generator(),
        len(X_train)//(batch_size *seq_length),
        validation_data=test_data_generator.frame_generator(),
        epochs = nb_epochs,
        validation_steps= len(X_test) // (batch_size *seq_length))

    lrcnm.model.save('my_model3.h5')

def main():
    model = 'lrcn'
    saved_model = None
    class_limit = None
    seq_length = 40
    batch_size = 1
    nb_epochs = 20

    if model is 'lrcn':
        image_shape = (360, 360, 3)

        train(seq_length, model, saved_model=saved_model, class_limit=class_limit,
            image_shape=image_shape, batch_size=batch_size, nb_epochs=nb_epochs)

if __name__ == '__main__':
    main()
