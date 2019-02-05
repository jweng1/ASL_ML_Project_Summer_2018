"""
Long-term Reccurent Convolutional Network (lrcn)

"""
from keras.layers import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D
import sys


class lrcnModel():
    def __init__(self, nb_classes, model, seq_length, saved_model = None, feat_length = 2040):

        self.nb_classes = nb_classes
        self.load_model = load_model
        self.seq_length = seq_length
        self.saved_model = saved_model

        if self.saved_model is not None:
            print('Loading model {}'.format(self.saved_model))
            self.model = load_model(self.saved_model)
        elif model == 'lrcn':
            print('Loading LRCN model.')
            self.input_shape = (seq_length, 360, 360, 3)
            self.model = self.lrcn()
        else:
            print('Unknown model')
            sys.exit()

        metrics = ['accuracy']
        optimizer =  Adam(lr=1e-3, decay=1e-4)
        #optimizer2 = RMSprop(lr=1e-3)

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

        print(self.model.summary())

    def lrcn(self):
        #CNN into RNN
        model = Sequential()
        #Define CNN
        model.add(TimeDistributed(Conv2D(8, (3, 3), strides=(2, 2), activation='sigmoid',
            padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(8, (3,3), init="he_normal", activation='sigmoid',
            padding='same')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(16, (3,3), padding='same', activation='sigmoid')))
        #model.add(TimeDistributed(Conv2D(16, (3,3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        #
        # model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        #
        # model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))


        model.add(TimeDistributed(Flatten()))
    
        #Define LSTM
        model.add(LSTM(4, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
