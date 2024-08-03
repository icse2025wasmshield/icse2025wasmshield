from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
import os
from time import perf_counter
import numpy as np
import torch
import wasmshield.utils
import random
from enum import Enum
import joblib
import seaborn as sns

def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

import matplotlib.pyplot as plt

class ConvRes(torch.nn.Module):
    def __init__(self, vec_size):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
        )
    def forward(self, x):
        return self.conv1(x)

class BatchType(Enum):
    obfuscated_formai=0
    optim_level_formai=1
    mutated_optim_levels_formai=2
    mutated_formai=3
    mutated_wasm_bench=4
    mutated_msr=5
    semantic_classification=6

class TrainableModel:

    def __init__(
        self, model, name, device
    ):
        if model is not None:
            self.model = model.to(device)
            self.history = defaultdict(list)
            self.test_history = defaultdict(list)
            self.epoch = 0
            self.name = name
        else:
            self.name=name
            print(f'Attempting to load model_name={self.name}')
            self.load()

    def save(self):
        torch.save(self.model,f'saving_models/models/{self.name}')
        joblib.dump(self.history,f'saving_models/history/{self.name}')
        joblib.dump(self.test_history,f'saving_models/test_history/{self.name}')
        joblib.dump(self.epoch,f'saving_models/epoch/{self.name}')

    def load(self):
        self.model = torch.load(f'saving_models/models/{self.name}')
        self.history = joblib.load(f'saving_models/history/{self.name}')
        self.test_history = joblib.load(f'saving_models/test_history/{self.name}')
        self.epoch = joblib.load(f'saving_models/epoch/{self.name}')

    def plot(self, figsize, ylim):

        plt.figure(figsize=figsize)

        plt.subplot(1,2,1)
        y,x = [],[]
        for idx,l in enumerate(self.history['overall']):
            for l2 in l:
                y.append(l2)
                x.append(idx+1)
        g = sns.lineplot(x=x,y=y, label=self.name+'_train_loss', )
        g.set(ylim=ylim)

        plt.subplot(1,2,2)
        for k in self.test_history.keys():
            if k in ('overall', 'time'):
                continue
            y,x = [],[]
            for idx,l in enumerate(self.test_history[k]):
                y.append(l)
                x.append(idx+1)
            g = sns.lineplot(x=x,y=y, label=k)
            g.set(ylim=ylim)

class Trainer:

    def __init__(
        self,
        trainable_model:TrainableModel,
        optimizer,
        preprocessor,
        formai_dataset,
        obfuscated_formai_dataset,
        wasm_bench_dataset,
        msr_dataset,
        device,
        training_preprocessor,
        semantic_dataset,
    ):
        
        self.trainable_model = trainable_model
        self.device = device
        self.training_preprocessor = training_preprocessor

        self.optimizer = optimizer

        self.formai_dataset_train, self.formai_dataset_test  = formai_dataset
        self.obfuscated_formai_dataset_train, self.obfuscated_formai_dataset_test = obfuscated_formai_dataset
        self.wasm_bench_dataset_train, self.wasm_bench_dataset_test = wasm_bench_dataset
        self.msr_dataset_train, self.msr_dataset_test = msr_dataset

        # (
        #     self.train_pairs_orig, 
        #     self.train_pairs_positive, 
        #     self.train_pairs_negative, 
        #     self.train_pairs_label,
        #     self.test_pairs_orig, 
        #     self.test_pairs_positive, 
        #     self.test_pairs_negative, 
        #     self.test_pairs_label
        # ) 
        
        
        self.semantic_X, self.semantic_y, self.semantic_train_idx, self.semantic_test_idx = semantic_dataset

        self.mutated_formai_folders = list(os.walk('compiled_datasets/mutated_formai/'))[0][1]
        self.mutated_wasm_bench_folders = list(os.walk('compiled_datasets/mutated_wasm_bench'))[0][1]
        self.mutated_msr_folders = list(os.walk('compiled_datasets/mutated_msr'))[0][1]

        self.preprocessor = preprocessor

        self.obfuscation_type_for_batch = None
        self.all_obfuscation_types = [
            'EncodeLiterals', 'Virtualize',
        ]
        self.obfuscation_type_for_batch_iter = iter(itertools.cycle([
            'AntiAliasAnalysis', 'AntiTaintAnalysis',
            'Copy', 'EncodeLiterals', 
            'Flatten', 'InitOpaque', 
            'Virtualize',
            # 'Random',
        ]))

    def train(
            
        self,
        startepoch,
        nbepochs,
        batch_size,
        batch_size_semantic,

        max_batch_size,

        batch_types:list[BatchType],
        criterion,

        criterion_semantic,
        semantic_mlp,

        n_reps_per_batch,
        n_batches_per_epoch=5,

        do_shuffle = True,
        is_for_simclr=True,
        do_eval=True,

        print_logs=True

    ):
        self.semantic_mlp=semantic_mlp

        for epoch in range(startepoch,nbepochs):

            time_begin_epoch = perf_counter()

            self.trainable_model.model = wasmshield.utils.set_torch_model_to_train(self.trainable_model.model, status=True).to(self.device)
            self.semantic_mlp = wasmshield.utils.set_torch_model_to_train(self.semantic_mlp, status=True).to(self.device)
            
            for batch_type in list(BatchType):
                self.trainable_model.history[batch_type].append([])
            self.trainable_model.history['time'].append([])
            self.trainable_model.history['overall'].append([])

        
            semantic_indices = np.arange(len(self.semantic_train_idx))
            if do_shuffle:
                random.shuffle(self.formai_dataset_train)
                random.shuffle(self.wasm_bench_dataset_train)
                random.shuffle(self.obfuscated_formai_dataset_train)
                random.shuffle(self.msr_dataset_train)
                random.shuffle(semantic_indices)

            formai_batches = wasmshield.utils.split_into_batches(self.formai_dataset_train,batch_size)
            wasm_bench_batches = wasmshield.utils.split_into_batches(self.wasm_bench_dataset_train,batch_size)
            msr_batches = wasmshield.utils.split_into_batches(self.msr_dataset_train,batch_size)
            obfuscated_formai_batches = wasmshield.utils.split_into_batches(self.obfuscated_formai_dataset_train,batch_size)
            semantic_batch_indices = wasmshield.utils.split_into_batches(semantic_indices,batch_size_semantic)

            batch_type_mapper={
                BatchType.obfuscated_formai : obfuscated_formai_batches,
                BatchType.mutated_formai : formai_batches,
                BatchType.optim_level_formai : formai_batches,
                BatchType.mutated_optim_levels_formai : formai_batches,
                BatchType.mutated_wasm_bench : wasm_bench_batches,
                BatchType.mutated_msr : msr_batches,
                BatchType.semantic_classification: semantic_batch_indices
            }

            batches_x =  [ 
                (bt,n_reps_per_batch[bt],batch_type_mapper[bt][0:n_batches_per_epoch]) for bt in batch_types
            ]

            batches_nb_epoch_reps = [x[1] for x in batches_x]
            batches_ = [x[2] for x in batches_x]
            batches_types = [x[0] for x in batches_x]
            batches_idx = {x:y for x,y in zip(batches_types, list(range(len(batches_))))}

            progress = [ 0 for _ in range(len(batches_)) ]
            N = max([len(b) for b in batches_])

            batch_idx=0
            batch_id =0

            while progress[0]!=0 or (batch_id==0):

                loss_ = 0

                for batch_type,nb_epoch_rep in zip(batches_types,batches_nb_epoch_reps):

                    if n_reps_per_batch[batch_type] == 0:
                        continue

                    batch_idx = batches_idx[batch_type]
                    # print(batch_type)

                    for i_epoch_rep in range(nb_epoch_rep):

                        self.obfuscation_type_for_batch = None

                        batch = batches_[batch_idx][progress[batch_idx]]

                        random.shuffle(batch)

                        sub_loss = 0
                        sub_n= 0

                        for sub_batch in divide_chunks(batch, max_batch_size):
                            sub_n+=1

                            x0,x1 = self.batch_processor(sub_batch,batch_type, is_for_simclr)

                            if batch_type == BatchType.semantic_classification:

                                z0 = self.trainable_model.model.backbone(x0,)
                                del x0

                                pred = self.semantic_mlp(z0)
                                del z0
                                loss = criterion_semantic(pred, torch.tensor(self.semantic_y[sub_batch], dtype=torch.long, device=self.device) )
                                loss.backward()
                                sub_loss += loss.item()

                                # pred = np.array(pred.detach().cpu().numpy()).argmax(axis=1)
                                # from sklearn.metrics import classification_report
                                # print(classification_report(self.semantic_y[batch], pred))
                                del pred
                                

                            elif is_for_simclr:

                                z0 = self.trainable_model.model(x0,)
                                del x0

                                z1 = self.trainable_model.model(x1,)
                                del x1
                            
                                loss = criterion(z0, z1,)
                                del z1
                                del z0

                                loss.backward()
                                sub_loss += loss.item()

                            self.trainable_model.history[batch_type][epoch].append(loss.item())
                            
                            # if batch_type == BatchType.semantic_classification:
                            #     progress[batch_idx] = (progress[batch_idx]+1)%len(batches_[batch_idx])
                            
                            # print(i_epoch_rep, loss.item())

                        loss_+=sub_loss/sub_n
                    
                    # if batch_type != BatchType.semantic_classification:
                    print(
                        batch_type,
                        progress[batch_idx],
                        len(batches_[batch_idx]),
                        # (batches_[batch_idx]),
                        # batches_,
                    )
                    progress[batch_idx] = (progress[batch_idx]+1)%len(batches_[batch_idx])
                
                if print_logs:
                    print('Stepping backward.')
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_div = sum(batches_nb_epoch_reps)
                if print_logs:
                    print('| epoch =', epoch+1, '| batch =', (batch_id+1), f"| loss = {(loss_/loss_div):.5f} |")
                    print('')
                self.trainable_model.history['overall'][epoch].append(loss_/loss_div)
                loss_ = 0
                batch_id+=1

            self.trainable_model.history['time'][epoch].append(perf_counter()-time_begin_epoch)
            
            if do_eval:
                self.eval(
                    criterion,
                    criterion_semantic,
                    batch_size,
                    batch_size_semantic,
                    n_reps_per_batch,
                    max_batch_size
                )

            self.trainable_model.epoch += 1

    def eval(self, criterion, criterion_semantic, batch_size, batch_size_semantic, n_reps_per_batch, max_batch_size):

        self.trainable_model.model = wasmshield.utils.set_torch_model_to_train(self.trainable_model.model, status=False)
        self.semantic_mlp = wasmshield.utils.set_torch_model_to_train(self.semantic_mlp, status=False)

        self.optimizer.zero_grad()
        
        random.shuffle(self.formai_dataset_test)
        random.shuffle(self.wasm_bench_dataset_test)
        random.shuffle(self.obfuscated_formai_dataset_test)
        semantic_indices_test = np.arange(len(self.semantic_test_idx))
        random.shuffle(semantic_indices_test)
        random.shuffle(self.msr_dataset_test)

        formai_batches_test = wasmshield.utils.split_into_batches(self.formai_dataset_test,batch_size)

        wasm_bench_dataset_batches_test = wasmshield.utils.split_into_batches(self.wasm_bench_dataset_test,batch_size)
        msr_dataset_batches_test = wasmshield.utils.split_into_batches(self.msr_dataset_test,batch_size)

        obfuscated_formai_dataset_batches_test = wasmshield.utils.split_into_batches(self.obfuscated_formai_dataset_test,batch_size)

        semantic_indices_batches_test = wasmshield.utils.split_into_batches(semantic_indices_test,batch_size_semantic)

        batch_type_mapper={
            BatchType.obfuscated_formai : obfuscated_formai_dataset_batches_test,
            BatchType.mutated_formai : formai_batches_test,
            BatchType.optim_level_formai : formai_batches_test,
            BatchType.mutated_optim_levels_formai : formai_batches_test,
            BatchType.mutated_wasm_bench : wasm_bench_dataset_batches_test,
            BatchType.mutated_msr : msr_dataset_batches_test,
            BatchType.semantic_classification : semantic_indices_batches_test,
        }

        batch_types = list(BatchType)

        tests = [
            [(bt,x) for x in batch_type_mapper[bt]][0:3] for bt in batch_types if n_reps_per_batch[bt]>0
        ]

        for test in (tests):

            loss_ = 0
            n = 0

            for (batch_type, batch) in (test):

                sub_loss_ = 0
                sub_n_ = 0

                random.shuffle(batch)

                for sub_batch in divide_chunks(batch, max_batch_size):

                    try:

                        for _ in range(n_reps_per_batch[batch_type]):

                            sub_n_ += 1

                            self.obfuscation_type_for_batch = None

                            x0,x1 = self.batch_processor(sub_batch, batch_type)

                        
                            if batch_type == BatchType.semantic_classification:

                                z0 = self.trainable_model.model.backbone(x0,)
                                del x0

                                pred = self.semantic_mlp(z0)
                                loss = criterion_semantic(pred, torch.tensor(self.semantic_y[sub_batch], dtype=torch.long, device=self.device) )

                                del pred
                                del z0

                            else:

                                z0 = self.trainable_model.model(x0,)
                                del x0

                                z1 = self.trainable_model.model(x1,)
                                del x1

                                loss = criterion(z0, z1,)

                                del z1
                                del z0

                            sub_loss_ += loss.item()
                            
                    
                    except (FileNotFoundError, ) as e:
                            continue
                
                loss_+= sub_loss_/(sub_n_ or 1)
                n+=1
            
            val_loss = loss_/(n or 1)
            self.trainable_model.test_history[batch_type].append(val_loss)

            print('\n Test_loss=', val_loss, 'Test_type=', batch_type)


    def preprocessor_task(self, file):
        return self.preprocessor(file)
    
    def batch_processor(self, batch, batch_type, is_for_simclr=True):
        
        view0 = []
        view1 = []

        if batch_type == BatchType.obfuscated_formai:

            for original_path, rec in batch:

                if self.obfuscation_type_for_batch is None:
                    self.obfuscation_type_for_batch = next(self.obfuscation_type_for_batch_iter)
                    # print(self.obfuscation_type_for_batch)
                if self.obfuscation_type_for_batch == 'Random':
                    obf_type = random.choice(self.all_obfuscation_types)
                    if is_for_simclr:
                        f1 = original_path
                        f2 = rec[obf_type]
                    else:
                        f2 = original_path # to orig
                        f1 = rec[obf_type] # obf
                else:
                    if is_for_simclr:
                        f1 = original_path
                        f2 = rec[self.obfuscation_type_for_batch]
                    else:
                        f2 = original_path # to orig
                        f1 = rec[self.obfuscation_type_for_batch] # obf
                

                x1 = (f1)
                x2 = (f2)

                view0.append(x1)
                view1.append(x2)

        elif batch_type == BatchType.optim_level_formai:

            for original_path, df in batch:

                files = df['path'][df['path'].str.contains('.wasm')].tolist()
                f1 = random.choice(files)
                if is_for_simclr:
                    f2 = f"compiled_datasets/formai/optimisation_levels/{original_path.split('.')[0].split('/')[-1]}_mutate_optimizer_emcc_O2.wasm"
                else:
                    files.remove(f1)
                    f2 = random.choice(files)

                x1 = (f1)
                x2 = (f2)
                view0.append(x1)
                view1.append(x2)

        elif batch_type == BatchType.mutated_optim_levels_formai:

            for original_path, df in batch:

                folder = random.choice(self.mutated_formai_folders)

                f1 = f"compiled_datasets/mutated_formai/{folder}/{original_path.split('.')[0].split('/')[-1]}_mutate_optimizer_emcc_O2.wasm"
                files = df['path'][df['path'].str.contains('.wasm')].tolist()
                f2 = random.choice(files)

                x1 = (f1)
                x2 = (f2)

                view0.append(x1)
                view1.append(x2)

        elif batch_type == BatchType.mutated_formai:

            for original_path, df in batch:

                folder = random.choice(self.mutated_formai_folders)
                f1 = f"compiled_datasets/mutated_formai/{folder}/{original_path.split('.')[0].split('/')[-1]}_mutate_optimizer_emcc_O2.wasm"
                f2 = f"compiled_datasets/formai/optimisation_levels/{original_path.split('.')[0].split('/')[-1]}_mutate_optimizer_emcc_O2.wasm"

                x1 = (f1)
                x2 = (f2)

                view0.append(x1)
                view1.append(x2)
        
        elif batch_type == BatchType.mutated_wasm_bench:

            for f_wasm_bench in batch:

                folder = random.choice(self.mutated_wasm_bench_folders)
                if is_for_simclr:
                    x = (f_wasm_bench)
                    y = (f"compiled_datasets/mutated_wasm_bench/{folder}/{f_wasm_bench.split('/')[-1]}")
                else:
                    y = (f_wasm_bench)
                    x = (f"compiled_datasets/mutated_wasm_bench/{folder}/{f_wasm_bench.split('/')[-1]}")
                
                view0.append(x)
                view1.append(y)

        elif batch_type == BatchType.mutated_msr:

            for f_wasm_bench in batch:

                folder = random.choice(self.mutated_msr_folders)
                if is_for_simclr:
                    x = (f_wasm_bench)
                    y = (f"compiled_datasets/mutated_msr/{folder}/{f_wasm_bench.split('/')[-1]}")
                else:
                    y = (f_wasm_bench)
                    x = (f"compiled_datasets/mutated_msr/{folder}/{f_wasm_bench.split('/')[-1]}")
                
                view0.append(x)
                view1.append(y)

        elif batch_type == BatchType.semantic_classification:
            view0.extend(self.semantic_X[batch])

        with ThreadPoolExecutor(max_workers=10) as pool:
            x0 = self.training_preprocessor(list(pool.map(self.preprocessor_task, view0)),self.device)
            x1 = self.training_preprocessor(list(pool.map(self.preprocessor_task, view1)),self.device)

        return x0, x1

