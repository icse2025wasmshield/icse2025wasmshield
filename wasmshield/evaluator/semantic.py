from time import perf_counter
import matplotlib.pyplot as plt 
import matplotlib

from wasmshield.models.base_handler import BaseHandler
plt.style.use('fivethirtyeight')
matplotlib.rcParams["figure.facecolor"] = 'white'
matplotlib.rcParams["axes.facecolor"] = 'white'

import os
from collections import defaultdict
import itertools
import random
import tqdm
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm
import numpy as np
import torch
import tqdm
from wasmshield.training.trainer import TrainableModel, divide_chunks
import joblib
from sklearn.base import BaseEstimator
import wasmshield.utils
import wasmshield.preprocessing
from sklearn.metrics import classification_report

from wasmshield.models.base_classifier import BaseClassifier

import pandas as pd
import seaborn as sns
import random
import numpy as np
import wasmshield.utils
from umap.umap_ import UMAP
import joblib


class SemanticEvaluator:
    def __init__(
        self,
    ):
        
        self.semantic_X=joblib.load('evaluation_logs/semantic_X')
        self.train_idx =joblib.load('evaluation_logs/semantic_train_idx')
        self.test_idx=joblib.load('evaluation_logs/semantic_test_idx')
        self.semantic_y=joblib.load('evaluation_logs/semantic_y')

        self.train_X = self.semantic_X[self.train_idx]
        self.train_y = self.semantic_y[self.train_idx]

        self.test_X = self.semantic_X[self.test_idx]
        self.test_y = self.semantic_y[self.test_idx]

        val_train_X, val_train_y = [], []
        val_validation_X, val_validation_y = [], []

        for y_label in set(self.semantic_y[self.train_idx]):
            mask = self.semantic_y[self.train_idx] == y_label
            x_for_label = self.semantic_X[self.train_idx][mask]
            y_for_label = self.semantic_y[self.train_idx][mask]
            
            idx = np.arange(len(x_for_label))
            random.shuffle(idx)

            p = 0.2
            n = int(len(x_for_label)*p)

            val_train_idx, val_validation_idx = idx[n:], idx[0:n]

            val_train_X.extend(x_for_label[val_train_idx])
            val_validation_X.extend(x_for_label[val_validation_idx])

            val_train_y.extend(y_for_label[val_train_idx])
            val_validation_y.extend(y_for_label[val_validation_idx])

        self.val_train_X = np.array(val_train_X)
        self.val_train_y = np.array(val_train_y)
        self.val_validation_X = np.array(val_validation_X)
        self.val_validation_y = np.array(val_validation_y)


    def train_many_models(
        self, 
        classifier_models:list[BaseClassifier],
    ):
        p = perf_counter()
        X_train = np.array(self.train_X)
        y_train = np.array(self.train_y)
        p = perf_counter() - p

        times = []
        for model in classifier_models:
            time = model.train_classifier(X_train, y_train)
            times.append(time)
        return {
            'training_times':times,
        }
    
    def eval_many_models(
        self, classifier_models:list[BaseClassifier],
    ):
        
        pred_times = []
        accuracies = []
        f1_scores = []
        overall_results_list = []

        for classifier_model in classifier_models:
            pred_time, y_pred = classifier_model.predict_from_files(
                files=self.test_X,
            )
            overall_results = classification_report(
                self.test_y, y_pred, output_dict=True
            )
            pred_times.append(pred_time)
            accuracies.append(
                overall_results['accuracy']
            )
            f1_scores.append(
                overall_results['weighted avg']['f1-score']
            )
            overall_results_list.append(overall_results)

        return {
            'pred_times':pred_times,
            'accuracies':accuracies,
            'f1_scores':f1_scores,
            'overall_results_list':overall_results_list,
        }
    
