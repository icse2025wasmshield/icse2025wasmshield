import torch
import torch.nn as nn
import subprocess
import random, os
import numpy as np
from tqdm import tqdm
import joblib

def load_datasets(
    ignore_those_that_dont_have_graphs=False
):

    keep_existing_files = lambda l : [f for f in l if os.path.isfile(f)]
    train_set, test_set = (joblib.load('metadata/formai_train_set')), (joblib.load('metadata/formai_test_set'))
    wasm_bench_train_set, wasm_bench_test_set = keep_existing_files(joblib.load('metadata/wasm_bench_train_set')), keep_existing_files(joblib.load('metadata/wasm_bench_formai_test_set'))
    msr_train_set, msr_test_set = keep_existing_files(joblib.load('metadata/train_msr_dataset')), keep_existing_files(joblib.load('metadata/test_msr_dataset'))
    clear_test_set = [x[0].split('.')[0].split('/')[-1] for x in test_set]
    clear_train_set = [x[0].split('.')[0].split('/')[-1]  for x in train_set]
    
    def obf_set_maker(clear_set, obfuscated_dataset_orig_path, ext='.c', add=""):
        _,transforms_used,_ = list(os.walk(obfuscated_dataset_orig_path))[0]
        obf_set = []
        for file in tqdm(clear_set):
            rec = {}
            orig = f"compiled_datasets/formai/optimisation_levels/{file}_mutate_optimizer_emcc_O2.wasm"
            for transf in transforms_used:
                path = obfuscated_dataset_orig_path+'/'+transf+'/'+file+ext
                if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
                    continue
                obfuscated_dataset_transform_path = add+f"{obfuscated_dataset_orig_path}"+'/'+transf+'/'+file+".wasm"
                if os.path.exists(obfuscated_dataset_transform_path):
                    rec[transf]= obfuscated_dataset_transform_path
            if len(list(rec.keys()))!=7:
                continue
            obf_set.append((orig,rec))
        return obf_set
    
    obfuscated_dataset_orig_path = "datasets/obfuscated_formai"
    obfuscated_train_set = obf_set_maker(clear_train_set, obfuscated_dataset_orig_path, add="compiled_")
    obfuscated_test_set = obf_set_maker(clear_test_set, obfuscated_dataset_orig_path, add="compiled_")

    llvm_obfuscated_dataset_orig_path = "compiled_datasets/llvm_formai"
    llvm_obfuscated_train_set = obf_set_maker(clear_train_set, llvm_obfuscated_dataset_orig_path, ext='.wasm')
    llvm_obfuscated_test_set = obf_set_maker(clear_test_set, llvm_obfuscated_dataset_orig_path, ext='.wasm')

    

    if ignore_those_that_dont_have_graphs:

        processed_files = set([
            x.replace('.joblib','').replace('$ç','/')
            for x in
            list(os.walk("/Users/walidthekraken/Documents/uqam/WALISM/datasets/wasm-shield-graphs-2/"))[0][2]
        ])

        mutated_formai_folders = list(os.walk('compiled_datasets/mutated_formai/'))[0][1]
        mutated_wasm_bench_folders = list(os.walk('compiled_datasets/mutated_wasm_bench'))[0][1]
        obfuscation_type_for_batch_iter = [
            'AntiAliasAnalysis', 'AntiTaintAnalysis',
            'Copy', 'EncodeLiterals', 
            'Flatten', 'InitOpaque', 
            'Virtualize',
        ]
        for wasm_bench_set in [
            wasm_bench_train_set, wasm_bench_test_set
        ]:
            wasm_bench_set_copy = wasm_bench_set.copy()
            for f_wasm_bench in tqdm(wasm_bench_set_copy):
                if f_wasm_bench not in processed_files:
                    wasm_bench_set.remove(f_wasm_bench)
                    continue
                for folder in mutated_wasm_bench_folders:
                    f_mutated_wasm_bench = (f"compiled_datasets/mutated_wasm_bench/{folder}/{f_wasm_bench.split('/')[-1]}")
                    if f_mutated_wasm_bench not in processed_files:
                        wasm_bench_set.remove(f_wasm_bench)
                        break
        for formai_set in [
            train_set, test_set
        ]:
            formai_set_copy = formai_set.copy()
            for f_formai_set in tqdm(formai_set_copy):
                formai_orginal_path, df = f_formai_set
                if not all(
                    [
                        x in processed_files
                        for x in
                        df['path'][df['path'].str.contains('.wasm')].tolist()
                    ]
                ):
                    formai_set.remove((f_formai_set))
                    continue
                for folder in mutated_formai_folders:
                    f2 = f"compiled_datasets/mutated_formai/{folder}/{formai_orginal_path.split('.')[0].split('/')[-1]}_mutate_optimizer_emcc_O2.wasm"
                    if f2 not in processed_files:
                        formai_set.remove(f_formai_set)
                        break

        for obf_set in [
            obfuscated_train_set, obfuscated_test_set
        ]:
            obf_set_copy = obf_set.copy()
            for dat in tqdm(obf_set_copy):
                _, rec = dat
                if not all(
                    [
                        f in processed_files
                        for f in rec.values()
                    ]
                ):    
                    obf_set.remove(dat)         

    return train_set, test_set, wasm_bench_train_set, wasm_bench_test_set, obfuscated_train_set, obfuscated_test_set, msr_train_set, msr_test_set, llvm_obfuscated_train_set, llvm_obfuscated_test_set

def set_torch_model_to_train(model: nn.Module, status: bool):
    model = model.train(mode=status)
    for param in model.parameters():
        param.requires_grad = status
    return model

def mutate_c_flags(filename, flag):
    # x in ['0', '1', '2', '3', 'z', 's']

    cfile_name = 'cool.c'
    hfile_name = 'cool.h'

    wasm_file_name = filename.split('.')[0]+'_OXYX.wasm'
    subprocess.run(['wasm2c', '-o', cfile_name, filename], capture_output=True).stderr

    wasm_file_name_x = wasm_file_name.replace('XYX',flag)
    subprocess.run(['emcc', f'-O{flag}', cfile_name, 'main.c', '-o', wasm_file_name_x], capture_output=True).stderr

    os.remove(cfile_name)
    os.remove(hfile_name)

    return wasm_file_name_x

def mutate_optimizer_binrayen(file,mode):
    import subprocess
    output = file.replace('.wasm', '_binaryen.wasm')
    subprocess.run(['wasm-opt', file, '-o', output, f'-O{mode}', '-c', '-all'])
    return output
    
def mutate_and_preprocess(filename, outputname, preprocess, nb_mutations, depth=1, remove=True, optimizer_mode=False, **preprocess_kwargs) -> list[np.array]:
    i = 0
    original_filename = filename
    images  : list[np.array] = []
    images.append(preprocess(filename, **preprocess_kwargs))
    while i<nb_mutations:
        d = 0
        newoutputname = outputname#.replace('NAMETAG', str(seed))
        while d<depth:
            seed = random.randint(0,15204560000)
            newoutputname = original_filename.split('.')[0] + str(d) + '.wasm'
            output = subprocess.run(['wasm-tools', 'mutate', filename, '--seed', str(seed), '-o', newoutputname, '--preserve-semantics'], capture_output=True).stderr
            if output and "There are not applicable mutations for the input Wasm module." in output.decode():
                # print("err", output)
                continue
            if remove and filename != original_filename:
                os.remove(filename)
            filename = newoutputname
            # print(filename)
            d+=1
        if optimizer_mode:
            filename_k = mutate_optimizer_binrayen(filename, random.choice([
                '0', 
                '1', '2', '3', 
                '4', 'z', 's']))
            if remove  and filename != original_filename:
                os.remove(filename)
            filename = filename_k
        # print('created =',newoutputname)
        image = preprocess(filename, **preprocess_kwargs)
        images.append(image)
        if remove  and filename != original_filename:
            os.remove(filename)
        i+=1
    return images


def mutate(filename, outputname):
    while True:
        seed = random.randint(0,15204560000)
        output = subprocess.run(['/opt/homebrew/bin/wasm-tools', 'mutate', filename, '-o', outputname, '--preserve-semantics', '--seed', str(seed)], capture_output=True).stderr
        if output and "There are not applicable mutations for the input Wasm module." in output.decode():
            continue
        break
    
def get_mutated_at_depth_modulo(
    filenames,
    in_folder,
    depth,
    modulo,
    delete_past_files=True,
    inject_random_chars=False,
    disable=False,
):

    import glob
    import uuid
    
    if delete_past_files:
        files = glob.glob(f'{in_folder}*.wasm')
        for f in files:
            os.remove(f)

    results = {}
    results[0] = filenames
    
    for filename in tqdm(filenames, disable=disable):
        last_filename = filename
        for d in range(1, depth+1):
            mutated_filename = in_folder + last_filename.split('/')[-1].split('.wasm')[0].split('$&')[0]+f'$&{d}'+(
                str(uuid.uuid4()) if inject_random_chars else ''
            )+'.wasm'
            mutate(last_filename, mutated_filename)
            if last_filename != filename and (d-1)%modulo!=1:
                # print('removed')
                os.remove(last_filename)
            if d%modulo==1:
                if results.get(d) is None:
                    results[d] = []
                results[d].append(mutated_filename)
            last_filename = mutated_filename
        if last_filename != filename and (d)%modulo!=1:
                os.remove(last_filename)
                # print('removed2')
    return results

def get_mutations_for_depths(name, list_of_depths, model, device, preprocess, **preprocess_kwargs):
    images = []
    images = images + mutate_and_preprocess( filename = name , outputname = 'test/'+'NAMETAG.wasm', nb_mutations = 0, depth=0, preprocess=preprocess, **preprocess_kwargs )
    for depth in tqdm(list_of_depths):
        images = images + [mutate_and_preprocess( filename = name , outputname = 'test/'+'NAMETAG.wasm', nb_mutations = 1, depth=depth, preprocess=preprocess, **preprocess_kwargs )[-1]]
        
    input = torch.tensor(np.array([x for x in images]), device=device, dtype=torch.float32).to(device)
    output = model(input).cpu().tolist()
    return output

def get_vectors(images, model, device):
    input = torch.tensor(np.array([x for x in images]), device=device, dtype=torch.float32).to(device)
    output = model(input).cpu().numpy()
    return output

def split_into_batches(l, batchsize):
    return [l[i:i + batchsize] for i in range(0, len(l), batchsize) if len(l[i:i + batchsize])==batchsize]


def fill_files_for_sizes(files_names_list:list[str], nb_files_to_get:int, files_dict:dict):

    while any([len(x[0])<nb_files_to_get for x in files_dict.values()]):

        file = random.choice(files_names_list)
        s = os.path.getsize(file)

        for files_dict_group, (files_dict_list, files_dict_lb, files_dict_ub) in files_dict.items():
            if files_dict_lb<=s<files_dict_ub:
                if len(files_dict_list)<nb_files_to_get:
                    files_dict_list.append(file)
                    break
                else:
                    files_names_list.remove(file)
                  