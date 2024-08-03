from collections import defaultdict
import itertools
import random
import matplotlib.pyplot as plt

from wasmshield.models.base_handler import BaseHandler 
plt.style.use('fivethirtyeight')

import sklearn.linear_model
import numpy as np
import torch
import tqdm
from wasmshield.training.trainer import TrainableModel, divide_chunks
import joblib
from sklearn.base import BaseEstimator
import wasmshield.utils
import wasmshield.preprocessing
import wasmshield.ennemies.jabberwock_pairs

import wasmshield.evaluator.transformation_detection
import wasmshield.training.trainer
import wasmshield.utils
import wasmshield.preprocessing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
device = 'mps'

import itertools
import os
import random
import joblib, tqdm
from collections import defaultdict
from enum import Enum

class transform_types(Enum):
    obfuscation = 'obfuscation'
    obfuscation_llvm = 'obfuscation_llvm'
    optim_levels = 'optim_levels'
    mutation_levels = 'mutation_levels'


class TransformsEvaluator:

    def __init__(
        self,
        device,
        n_train,
        n_test,
    ):

        (
            train_set, test_set, 
            wasm_bench_train_set, wasm_bench_test_set, 
            obfuscated_train_set, obfuscated_test_set, 
            msr_train_set, msr_test_set, 
            llvm_obfuscated_train_set, llvm_obfuscated_test_set
        ) = wasmshield.utils.load_datasets()

        self.device = device
        self.n_train = n_train
        self.n_test = n_test

        try:

            self.test_new_pairs_f = joblib.load('evaluation_logs/test_new_pairs_f')
            self.posi_test_new_pairs = joblib.load('evaluation_logs/posi_test_new_pairs')
            self.nega_test_new_pairs = joblib.load('evaluation_logs/nega_test_new_pairs')

        except:

            self.test_new_pairs_f = defaultdict(list)
            self.posi_test_new_pairs = defaultdict(list)
            self.nega_test_new_pairs = defaultdict(list)

            self.fill_pairs(
                s_list=[
                    (
                        transform_types.obfuscation_llvm,
                        llvm_obfuscated_train_set,
                        llvm_obfuscated_test_set,
                    )
                ],
                train_f_pairs=defaultdict(list),
                train_posi_pairs=defaultdict(list),
                train_nega_pairs=defaultdict(list),
                test_f_pairs=self.test_new_pairs_f,
                test_posi_pairs=self.posi_test_new_pairs,
                test_nega_pairs=self.nega_test_new_pairs,
            )
            
            joblib.dump(self.test_new_pairs_f,'evaluation_logs/test_new_pairs_f')
            joblib.dump(self.posi_test_new_pairs,'evaluation_logs/posi_test_new_pairs')
            joblib.dump(self.nega_test_new_pairs,'evaluation_logs/nega_test_new_pairs')

        try:

            self.test_pairs_f = joblib.load('evaluation_logs/test_pairs_f')
            self.posi_test_pairs = joblib.load('evaluation_logs/posi_test_pairs')
            self.nega_test_pairs = joblib.load('evaluation_logs/nega_test_pairs')

            self.train_pairs_f = joblib.load('evaluation_logs/train_pairs_f')
            self.posi_train_pairs = joblib.load('evaluation_logs/posi_train_pairs')
            self.nega_train_pairs = joblib.load('evaluation_logs/nega_train_pairs')
            
            print('Preloaded pairs')

        except:

            print('Loading pairs')

            self.test_pairs_f = defaultdict(list)
            self.posi_test_pairs = defaultdict(list)
            self.nega_test_pairs = defaultdict(list)

            self.train_pairs_f = defaultdict(list)
            self.posi_train_pairs = defaultdict(list)
            self.nega_train_pairs = defaultdict(list)

            self.fill_pairs(
                s_list=[
                    (
                        transform_types.obfuscation,
                        obfuscated_train_set,
                        obfuscated_test_set,
                    ),
                    (
                        transform_types.optim_levels,
                        train_set,
                        test_set,
                    ),
                    (
                        transform_types.mutation_levels,
                        (train_set,wasm_bench_train_set,msr_train_set),
                        (test_set,wasm_bench_test_set,msr_test_set),
                    ),
                ],
                train_f_pairs=self.train_pairs_f,
                train_posi_pairs=self.posi_train_pairs,
                train_nega_pairs=self.nega_train_pairs,
                test_f_pairs=self.test_pairs_f,
                test_posi_pairs=self.posi_test_pairs,
                test_nega_pairs=self.nega_test_pairs,
            )
            
            joblib.dump(self.test_pairs_f,'evaluation_logs/test_pairs_f')
            joblib.dump(self.posi_test_pairs,'evaluation_logs/posi_test_pairs')
            joblib.dump(self.nega_test_pairs,'evaluation_logs/nega_test_pairs')
            
            joblib.dump(self.train_pairs_f,'evaluation_logs/train_pairs_f')
            joblib.dump(self.posi_train_pairs,'evaluation_logs/posi_train_pairs')
            joblib.dump(self.nega_train_pairs,'evaluation_logs/nega_train_pairs')

        # try:

        #     raise Exception

        #     (
        #         self.X_train_jw, 
        #         self.y_train_jw,
        #         self.X_test_jw,
        #         self.y_test_jw
        #     ) = (
        #         joblib.load("evaluation_logs/transforms_X_train_jw"),
        #         joblib.load("evaluation_logs/transforms_y_train_jw"),
        #         joblib.load("evaluation_logs/transforms_X_test_jw"),
        #         joblib.load("evaluation_logs/transforms_y_test_jw")
        #     )

        #     print('Preloaded jw results')
        
        # except:

        #     print('Reloaing jw results')

        #     # print(len(self.train_pairs_f['mutated_Under_1KB']))
        #     # print(len(self.posi_train_pairs['mutated_Under_1KB']))
        #     # print(len(self.nega_train_pairs['mutated_Under_1KB']))

        #     (
        #         self.X_train_jw, 
        #         self.y_train_jw,
        #         self.X_test_jw,
        #         self.y_test_jw,
        #         self.X_test_new_jw,
        #         self.y_test_new_jw,
        #     ) = self.get_X_y(
        #         vector_getter=wasmshield.ennemies.jabberwock_pairs.get_vectors_jabberwock,
        #     )

        #     joblib.dump(self.X_train_jw,"evaluation_logs/transforms_X_train_jw")
        #     joblib.dump(self.y_train_jw,"evaluation_logs/transforms_y_train_jw")
        #     joblib.dump(self.X_test_jw,"evaluation_logs/transforms_X_test_jw")
        #     joblib.dump(self.y_test_jw,"evaluation_logs/transforms_y_test_jw")
        #     joblib.dump(self.X_test_new_jw,"evaluation_logs/transforms_X_test_new_jw")
        #     joblib.dump(self.y_test_new_jw,"evaluation_logs/transforms_y_test_new_jw")

    def fill_pairs(
        self, 
        s_list, 
        train_f_pairs, 
        train_posi_pairs,
        train_nega_pairs,
        test_f_pairs, 
        test_posi_pairs,
        test_nega_pairs,
    ):
        for s_type, s_train, s_test in s_list:
            for s, f_pairs, posi_pairs, nega_pairs, n in [
                (
                    s_train,
                    train_f_pairs,
                    train_posi_pairs,
                    train_nega_pairs,
                    self.n_train,
                ),
                (
                    s_test,
                    test_f_pairs,
                    test_posi_pairs,
                    test_nega_pairs,
                    self.n_test
                ),
            ]:
                self.transform_processor(n, s, f_pairs, posi_pairs, nega_pairs, s_type)

    def get_pairs_vectors(self, pair, handler:BaseHandler, chunksize=512,):
        k = {}
        for t, li in tqdm.tqdm(list(pair.items())):
            k[t] = []
            for chunk in divide_chunks(li,chunksize):
                k[t].extend(handler.get_vectors_from_files(chunk))
        return k
    
    def get_X_y(self, handler:BaseHandler):

        train_pairs_f_v,posi_train_pairs_v,nega_train_pairs_v = (
            self.get_pairs_vectors(self.train_pairs_f, handler),
            self.get_pairs_vectors(self.posi_train_pairs, handler),
            self.get_pairs_vectors(self.nega_train_pairs, handler),
        )
        
        test_pairs_f_v,posi_test_pairs_v,nega_test_pairs_v = (
            self.get_pairs_vectors(self.test_pairs_f, handler),
            self.get_pairs_vectors(self.posi_test_pairs, handler),
            self.get_pairs_vectors(self.nega_test_pairs, handler),
        )

        test_new_pairs_f_v,posi_test_new_pairs_v,nega_test_new_pairs_v = (
            self.get_pairs_vectors(self.test_new_pairs_f, handler),
            self.get_pairs_vectors(self.posi_test_new_pairs, handler),
            self.get_pairs_vectors(self.nega_test_new_pairs, handler),
        )

        X_train = []
        y_train = []
        for t in train_pairs_f_v.keys():
            x_posi = np.concatenate([train_pairs_f_v[t], posi_train_pairs_v[t]], axis=1)
            x_nega = np.concatenate([train_pairs_f_v[t], nega_train_pairs_v[t]], axis=1)
            X_train.extend(x_posi)
            y_train.extend([1 for _ in range(len(x_posi))])
            X_train.extend(x_nega)
            y_train.extend([0 for _ in range(len(x_nega))])
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = defaultdict(list)
        y_test = defaultdict(list)
        for t in test_pairs_f_v.keys():
            x_posi = np.concatenate([test_pairs_f_v[t], posi_test_pairs_v[t]], axis=1)
            x_nega = np.concatenate([test_pairs_f_v[t], nega_test_pairs_v[t]], axis=1)
            X_test[t].extend(x_posi)
            y_test[t].extend([1 for _ in range(len(x_posi))])
            X_test[t].extend(x_nega)
            y_test[t].extend([0 for _ in range(len(x_nega))])

        X_test_new = defaultdict(list)
        y_test_new = defaultdict(list)
        for t in test_new_pairs_f_v.keys():
            x_posi = np.concatenate([test_new_pairs_f_v[t], posi_test_new_pairs_v[t]], axis=1)
            x_nega = np.concatenate([test_new_pairs_f_v[t], nega_test_new_pairs_v[t]], axis=1)
            X_test_new[t].extend(x_posi)
            y_test_new[t].extend([1 for _ in range(len(x_posi))])
            X_test_new[t].extend(x_nega)
            y_test_new[t].extend([0 for _ in range(len(x_nega))])

        return X_train, y_train, X_test, y_test, X_test_new, y_test_new
            

    def transform_processor(self, n, s, f_pairs, posi_pairs, nega_pairs, s_type):

        mutated_conditions = {
            'mutated_Under_1KB': lambda x : 10**3>os.path.getsize(x),
            'mutated_1KB_and_100KB': lambda x : 10**5>os.path.getsize(x)>=10**3,
            'mutated_100KB_and_1MB': lambda x : 10**6>os.path.getsize(x)>=10**5,
            'mutated_1MB_and_100MB': lambda x : 10**8>os.path.getsize(x)>=10**6,
            # 'mutated_10MB': lambda x : os.path.getsize(x)>=10**7,
        }

        print(s_type, n)

        if s_type in (transform_types.obfuscation, transform_types.obfuscation_llvm):
            for fi, fi_rec in tqdm.tqdm(sorted(s, reverse=True, key=lambda x:len(list(x[1].keys())))[:n]):
                for o_type, ofi in fi_rec.items():
                    while True:
                        nfi, nfi_rec = random.choice(s)
                        if nfi == fi:
                            continue
                        elif nfi_rec.get(o_type) is None:
                            continue
                        else:
                            break
                    f_pairs['obfuscation_'+o_type].append(fi)
                    posi_pairs['obfuscation_'+o_type].append(ofi)
                    nega_pairs['obfuscation_'+o_type].append(nfi_rec[o_type])

        elif s_type is transform_types.optim_levels:
            for f_orig, df in tqdm.tqdm(s[:n]):
                l = [fname for fname in df.path.tolist() if '.wasm' in fname]
                combinations = list(itertools.combinations(l, 2))
                for f1, f2 in (combinations):
                    o1 = f1.split('_')[-1].split('.')[0]
                    o2 = f2.split('_')[-1].split('.')[0]
                    o_type = f'{o1}_{o2}'
                    while True:
                        nf_orig, ndf = random.choice(s)
                        if f_orig == nf_orig:
                            continue
                        else:
                            nf_orig_bin = [fname for fname in ndf.path.tolist() if '.wasm' in fname][0]
                            nf_orig_bin = nf_orig_bin.replace(nf_orig_bin.split('_')[-1], f"{o2}.wasm")
                            # print(nf_orig_bin)
                            if not os.path.exists(nf_orig_bin):
                                continue
                            break
                    f_pairs[o_type].append(f1)
                    posi_pairs[o_type].append(f2)
                    nega_pairs[o_type].append(nf_orig_bin)
        
        elif s_type is transform_types.mutation_levels:

            s_formai, s_wasmbench, s_msr = s

            s_formai = [x[0] for x in s_formai]
            mutated_path_formai = 'compiled_datasets/mutated_formai/'
            clear_path_formai = 'compiled_datasets/formai/optimisation_levels/'
            folder_formai = list(os.walk(mutated_path_formai))[0][1][-1]
            s_formai = [
                (
                    clear_path_formai+x.split('/')[-1].replace('.c','_mutate_optimizer_emcc_O2.wasm'),
                    mutated_path_formai+folder_formai+'/'+x.split('/')[-1].replace('.c','_mutate_optimizer_emcc_O2.wasm'),

                ) for x in s_formai
            ]

            mutated_path_wasmbench = 'compiled_datasets/mutated_wasm_bench/'
            folder_wasmbench = list(os.walk(mutated_path_wasmbench))[0][1][-1]
            s_wasmbench = [
                (
                    x,
                    mutated_path_wasmbench+folder_wasmbench+'/'+x.split('/')[-1],

                ) for x in s_wasmbench
            ]

            mutated_path_msr = 'compiled_datasets/mutated_msr/'
            folder_msr = list(os.walk(mutated_path_msr))[0][1][-1]
            s_msr = [
                (
                    x,
                    mutated_path_msr+folder_msr+'/'+x.split('/')[-1],

                ) for x in s_msr
            ]

            for cond_name, cond in (mutated_conditions.items()):

                _s = s_wasmbench+s_formai+s_msr
                random.shuffle(_s)
                conded_origs = [x for x in _s if cond(x[0])][0:n]
                
                for orig in tqdm.tqdm(conded_origs):

                    f_mutated_path, f_clear_path = orig

                    while True:
                        f_mutated_path_nega, f_clear_path_nega = random.choice(conded_origs)
                        if f_clear_path_nega == f_clear_path:
                            continue
                        break

                    # f_nega_path = orig_nega
                    if not all([os.path.exists(f) for f in (f_clear_path, f_mutated_path, f_mutated_path_nega)]):
                        raise Exception(f'File not found {f_clear_path} {f_mutated_path} {f_mutated_path_nega}',)
                    
                    f_pairs[cond_name].append(f_clear_path)
                    posi_pairs[cond_name].append(f_mutated_path)
                    nega_pairs[cond_name].append(f_mutated_path_nega)

