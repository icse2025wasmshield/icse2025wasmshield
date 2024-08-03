
from time import perf_counter
from sklearn.naive_bayes import LabelBinarizer
from wasmshield.models.base_classifier import BaseClassifier
from wasmshield.models.base_handler import BaseHandler
import math
import numpy as np
from PIL import Image
import keras

import math
import wasmshield.ennemies.minos
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Dropout


def build_minos_cnn(size=(100,100), num_classes=2):
    model = Sequential()
    model.add(Reshape((*size, 1), input_shape=(size[0]*size[1], )))
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(*size,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class MinosHandler(BaseHandler):

    def __init__(self, model):
        self.model = model
        self.vectorizer = keras.Model(inputs = self.model.input, outputs = self.model.layers[-2].output)

    def preprocess_file(self, file:str, size:int=100):
        with open(file, 'rb') as f:
            bytes = f.read()
        return self.preprocess_bytes(bytes, size)
    
    def preprocess_files(self, files:list, size:int=100):
        return np.array([
            self.preprocess_file(f)[0]
            for f in files
        ])
    
    def preprocess_bytes(self, bytes:bytes, size:int=100):
        size = len(bytes)
        w = int(math.sqrt(size))
        if w > 0:
            h = int(size/w)
        img = Image.frombuffer('L', (w,h), bytes)
        img = img.resize((100, 100))
        return np.array([np.asarray(img).flatten()])
    
    def get_vector_from_preprocessed_file(self, preprocessed_file, device='mps'):
        k = np.array(preprocessed_file)
        k = self.vectorizer.predict(k, verbose = 0)
        return k
    
    def get_vectors_from_files(
        self, 
        files, 
        size:int=100, 
        max_workers=24,
        device='mps'
    ):
        k = np.array(
            [self.preprocess_file(f, size=size)[0] for f in files]
        )
        k = self.vectorizer.predict(k, verbose = 0)
        return k

class MinosClassifier(BaseClassifier):

    def __init__(self, model, classifier_model=None, handler_cls=MinosHandler):
        self.model = model
        self.handler = handler_cls(model)
        self.classifier_model = classifier_model

    def parse_minos_y(self, y):
        k = np.array(y)
        return k

    def predict_from_files(
        self, 
        files, 
        size:int=0, 
    ):
        t = perf_counter()
        images = self.handler.preprocess_files(files)
        k = self.model.predict(images, verbose = 0).argmax(axis=1)
        return (perf_counter()-t, self.parse_minos_y(k))
    
    def predict_proba_from_files(
        self, 
        files, 
        size:int=0, 
    ):
        t = perf_counter()
        images = self.handler.preprocess_files(files)
        k = self.model.predict(images, verbose = 0)
        return (perf_counter()-t, self.parse_minos_y(k))

    def train_classifier(
        self,
        X_train,
        y_train,
        size:int=0,
        epochs=20,
    ):
        
        t = perf_counter()
        X_train_minos = self.handler.preprocess_files(X_train)
        y_train_minos = np.zeros(
            (len(y_train), len(np.unique(y_train)))
        )
        for idx,y in enumerate(y_train):
            y_train_minos[idx,y]=1
        print(X_train_minos.shape, y_train_minos.shape)
        p = np.arange(0, len(X_train_minos))
        np.random.shuffle(p)
        X_train_minos, y_train_minos = X_train_minos[p], y_train_minos[p]
        self.model.fit(X_train_minos, y_train_minos, epochs=epochs)
        t = perf_counter() - t
        return t
        
class PreTrainedMinosClassifier(BaseClassifier):

    def __init__(self, model, classifier_model=None, handler_cls=MinosHandler):
        self.model = model
        self.handler = handler_cls(model)
        self.classifier_model = classifier_model

    def parse_minos_y(self, y):
        k = np.array(y)
        return 1-k

    def predict_from_files(
        self, 
        files, 
        size:int=0, 
    ):
        t = perf_counter()
        images = self.handler.preprocess_files(files)
        k = self.model.predict(images, verbose = 0).argmax(axis=1)
        return (perf_counter()-t, self.parse_minos_y(k))

    def train_classifier(
        self,
        X_train,
        y_train,
        size:int=0,
        epochs=20,
    ):
        return 0
        

