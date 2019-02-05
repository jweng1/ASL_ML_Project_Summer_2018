import numpy as np
from sklearn.model_selection import train_test_split

class KerasBatchGenerator(object):

    def __init__(self, X, y, batch_size, nb_classes = 59, seq_length=40, image_shape=(360, 360, 3)):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.nb_classes = nb_classes
        self.image_shape = image_shape
        self.current_index = 0


    def frame_generator(self):

        X = np.zeros((self.batch_size, self.seq_length, *self.image_shape))
        y = np.zeros((self.batch_size, self.nb_classes))
        
        #print(y.shape)
        #print(self.y.shape)
        while True:
            for i in range(self.batch_size):
                if self.current_index + self.seq_length >= len(self.X):
                    self.current_index = 0
                X[i] = self.X[self.current_index:self.current_index + self.seq_length]
                y[i] = self.y[i]

                self.current_index += 1

            yield X,y
