import abc
from time import perf_counter

import numpy as np
from wasmshield.models.base_handler import BaseHandler

class BaseClassifier():

    def __init__(self, model, classifier_model, handler_cls):
        self.model = model
        self.handler = handler_cls()
        self.classifier_model = classifier_model

    def predict_from_files(
        self, 
        files, 
        size:int=0, 
        max_workers=24,
    ):
        pass

    def predict_proba_from_files(
        self, 
        files, 
        size:int=0, 
        max_workers=24,
    ):
        pass

    def train_classifier(
        self,
        X_train,
        y_train,
        size:int=0,
        epochs=50,
    ):
        pass

class SklearnClassifier():
    def __init__(self, model, classifier_model, handler_cls):
        self.model = model
        self.handler:BaseHandler = handler_cls(model)
        self.classifier_model = classifier_model

    def predict_from_files(
        self, 
        files, 
        size:int=0, 
        max_workers=24,
    ):
        t = perf_counter()
        vectors = self.handler.get_vectors_from_files(files)
        preds = self.classifier_model.predict(vectors)
        return (perf_counter()-t), preds
    
    def predict_proba_from_files(
        self, 
        files, 
        size:int=0, 
        max_workers=24,
    ):
        t = perf_counter()
        vectors = self.handler.get_vectors_from_files(files)
        preds = self.classifier_model.predict_proba(vectors)
        return (perf_counter()-t), preds

    def train_classifier(
        self,
        X_train,
        y_train,
        size:int=0,
        epochs=50,
    ):
        t = perf_counter()
        vectors = self.handler.get_vectors_from_files(X_train)
        p = np.arange(0, len(X_train))
        np.random.shuffle(p)
        self.classifier_model.fit(vectors[p], y_train[p])
        return (perf_counter()-t)