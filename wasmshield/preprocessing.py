from concurrent.futures import ThreadPoolExecutor
import re
import subprocess
import pandas as pd
from array import array
import os
import math
import os
import numpy as np
import random
import joblib
import zlib

import PIL.Image
from PIL.Image import Resampling

import torch
import sexpdata
import tqdm

def ins_cleaner(i):
    i = i.replace('atomic.', '').split('.')[-1]
    return i

def parse_sexpr(s):
    return sexpdata.loads(s)

def parse_wasm_module(file, print_logs=False):
    import subprocess
    if print_logs:
        print_logs("Getting Text Version")
    res = subprocess.run([ 'wasm-tools' , 'print' , file], capture_output=True)
    s = res.stdout.decode().replace(';; label = ','').replace('(;','').replace(';)','')
    error = res.stderr.decode()
    if len(error)>0:
        # print(file)
        return None, None
    if print_logs:
        print_logs("Getting S-Expression")
    result = parse_sexpr(s)
    return result, s

def parse_varint(data, offset ):
    result = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            break
    return result, offset

def preprocess_image_one(file, size=128, use_pil=False, compress=False):
    return preprocess_image(
        file, size, group_mapping = { 
            0: 0, 1: 0, 2: 0, 3: 0, 7: 0, 
            10: 0, 8: 0, # CODE was 1
            11: 0, 12: 0, 4: 0,  5: 0, 6: 0, 9: 0, # was 2
        }, use_pil=use_pil, compress=compress
    )

def preprocess_image_one_from_bytes(file, size=128, use_pil=False, compress=False):
    return preprocess_image_from_bytes(
        file, size, group_mapping = { 
            0: 0, 1: 0, 2: 0, 3: 0, 7: 0, 
            10: 0, 8: 0, # CODE was 1
            11: 0, 12: 0, 4: 0,  5: 0, 6: 0, 9: 0, # was 2
        }, use_pil=use_pil, compress=compress
    )

def get_code_bytes(file):
    code_groups = {10,8}
    f = open(file, 'rb')
    code = b''
    wasm_data = f.read()
    offset = 8  # Start reading after the header
    while offset < len(wasm_data):
        section_id = wasm_data[offset]
        offset += 1
        section_length, offset = parse_varint(wasm_data, offset)
        section_data = wasm_data[offset:offset + section_length]
        offset += section_length
        if section_id not in code_groups:
            code += section_data
    f.close()
    return code


def preprocess_image(file,size=128, group_mapping = { 
    0: 0, 1: 0, 2: 0, 3: 0, 7: 0,
    10: 1, 8: 1, # CODE was 1
    11: 2, 12: 2, 4: 2,  5: 2, 6: 2, 9: 2, # was 2,
    }, use_pil=False, compress=False,) -> np.array:
    """
        WASM EN IMAGE
    """
    
    f = open(file, 'rb')
    imgs = []

    groups = {}
    for k in range(len(set(group_mapping.values()))):
        groups[k] = b''
    
    wasm_data = f.read()
    offset = 8  # Start reading after the header
    while offset < len(wasm_data):
        section_id = wasm_data[offset]
        offset += 1
        section_length, offset = parse_varint(wasm_data, offset)
        section_data = wasm_data[offset:offset + section_length]
        offset += section_length
        groups[group_mapping.get(section_id, 0)] += section_data
    f.close()
    
    for section in groups.values():
        if compress:
            section = zlib.compress(section)
        # ln = os.stat(file).st_size
        ln = len(section)
        width = max(int(math.pow(ln, 0.5)), 1)
        rem = ln % width
        # print(ln, width, rem)
        a = array('B')
        # a.fromfile(f, ln-rem)
        a.frombytes(section[:width*width])
        # print(a)
        # f.close()
        g = np.reshape(a,(int(len(a)/width), width))
        g = np.uint8(g)
        if not use_pil:
            h = np.resize(g, (size, size))
        else:
            h = np.array(PIL.Image.fromarray(g).resize((size, size), resample=Resampling.NEAREST), dtype=np.uint8)

        # h = h/255

        # h = h.reshape((-1, 100, 100, 1))
        imgs.append(h)
    
    return np.array(imgs, dtype=np.uint8)

def preprocess_image_from_bytes(
    wasm_data, 
    size=128, 
    group_mapping = { 
        0: 0, 1: 0, 2: 0, 3: 0, 7: 0,
        10: 1, 8: 1, # CODE was 1
        11: 2, 12: 2, 4: 2,  5: 2, 6: 2, 9: 2, # was 2,
        }, use_pil=False, compress=False,
    ) -> np.array:
    """
        WASM EN IMAGE
    """

    imgs = []

    groups = {}
    for k in range(len(set(group_mapping.values()))):
        groups[k] = b''
    
    offset = 8  # Start reading after the header
    while offset < len(wasm_data):
        section_id = wasm_data[offset]
        offset += 1
        section_length, offset = parse_varint(wasm_data, offset)
        section_data = wasm_data[offset:offset + section_length]
        offset += section_length
        groups[group_mapping.get(section_id, 0)] += section_data
    
    for section in groups.values():
        if compress:
            section = zlib.compress(section)
        # ln = os.stat(file).st_size
        ln = len(section)
        width = max(int(math.pow(ln, 0.5)), 1)
        rem = ln % width
        # print(ln, width, rem)
        a = array('B')
        # a.fromfile(f, ln-rem)
        a.frombytes(section[:width*width])
        # print(a)
        # f.close()
        g = np.reshape(a,(int(len(a)/width), width))
        g = np.uint8(g)
        if not use_pil:
            h = np.resize(g, (size, size))
        else:
            h = np.array(PIL.Image.fromarray(g).resize((size, size), resample=Resampling.NEAREST), dtype=np.uint8)

        # h = h/255

        # h = h.reshape((-1, 100, 100, 1))
        imgs.append(h)
    
    return np.array(imgs, dtype=np.uint8)

def preprocess_graph(
    file:str,
    save_graphs_in="/Users/walidthekraken/Documents/uqam/WALISM/datasets/wasm-shield-graphs-2/",
    print_logs=False
):

    save_file_in = save_graphs_in+file.replace('/','$ç')+'.joblib'
    if os.path.exists(save_file_in):
        try:
            return joblib.load(save_file_in)
        except (EOFError,KeyError):
            pass

    if print_logs:
        print(file, os.path.getsize(file)/1024)
        print("Generating Adj List")

    r = subprocess.run([
        'wassail', 'callgraph-adjlist', f"{file}", '/dev/stdout'
    ], capture_output=True, )
    stderr = r.stderr.decode().split('\n')
    r = r.stdout.decode().split('\n')

    adj_lists = [[], [],]
    edge_types = []
    if print_logs:
        print("Processing Adj List")
        print(r)
        print(stderr)
    for x in r:
        if len(x)==0:
            continue
        _from, _to, _type = x.split()
        adj_lists[0].append(int(_from))
        adj_lists[1].append(int(_to))
        edge_types.append(_type)
    functions_instructions = []
    function_ids = sorted(set(adj_lists[0]+adj_lists[1]))
    node_ids = {func_id:i for i,func_id in enumerate(function_ids)}

    def has_numbers(inputString):
        return bool(re.search(r'\d', inputString))
    
    def is_hex(s):
        try:
            int(s, 16)
            return True
        except ValueError:
            return False

    def token_cleaner(token:str):
        if (
            '$' in token 
            or "@" in token 
            or '=' in token
            or 'nan' == token
        ):
            return ''
        token = token.strip()
        token, *args = token.split(' ')
        *types, token = token.split('.')
        i = 1
        while len(types)>=i:
            if types[-i] in {'local', 'global', 'atomic', 'memory'}:
                token = f"{types[-i]}.{token}"
                i+=1
            else:
                break
        cleaner = lambda x, w : x.replace(f'{w}x', w)
        token = cleaner(token, 'load')
        token = cleaner(token, 'store')
        token = cleaner(token, 'convert')
        token = token.replace('8', '').replace('16', '').replace('32', '').replace('64', '')
        token = re.sub(r'_i(?![a-zA-Z])', r'', token)
        token = re.sub(r'_s(?![a-zA-Z])', r'', token)
        token = re.sub(r'_u(?![a-zA-Z])', r'', token)
        token = re.sub(r'_f(?![a-zA-Z])', r'', token)
        token = re.sub(r'_v(?![a-zA-Z])', r'', token)
        token = re.sub(r'_x(?![a-zA-Z])', r'', token)
        token = token.replace('-','')
        token = token.replace('+','')
        if (
            has_numbers(token) 
            or is_hex(token.replace('p',''))
            or len(token.replace('p',''))==0
        ):
            return ''
        return token

    # def run(args):
    #     file,func = args
    #     r = subprocess.run([
    #         'wassail', 'function-body', file, str(func),
    #     ], capture_output=True).stdout.decode().split('\n')
    #     lst = [token_cleaner(x) for x in r if len(x)>0]
    #     return [lst[i] for i in range(len(lst)) if i == 0 or lst[i] != lst[i-1]]

    if print_logs:
        print("Parsing Module")
    result, _ = parse_wasm_module(file, print_logs=print_logs)
    module = result[1:]
    
    def run(args):
        part = args
        r = [str(x) for x in part[2:] if ( not isinstance(x, list)) ]
        lst = [token_cleaner(x) for x in r if len(x)>0]
        lst = [x for x in lst if len(x)>0]
        return [lst[i] for i in range(len(lst)) if i == 0 or lst[i] != lst[i-1]]
    
    with ThreadPoolExecutor(max_workers=15) as pool:
        results = list(tqdm.tqdm(pool.map(run, [(func) for func in [part for part in module if str(part[0]) == 'func']])))
    
    # with ThreadPoolExecutor(max_workers=5) as pool:
    #     results = pool.map(run, [(file, func) for func in function_ids])

    for tokens in results:
        functions_instructions.append(tokens)

    to_return = ([
        [ node_ids[i] for i in adj_lists[0] ], [ node_ids[i] for i in adj_lists[1] ]
    ], edge_types, functions_instructions)

    joblib.dump(to_return,save_file_in)

    return to_return

def preprocess_text(
    file:str,
    save_graphs_in="/Users/walidthekraken/Documents/uqam/WALISM/datasets/wasm-shield-graphs-2/",
    print_logs=False
):


    save_file_in = save_graphs_in+file.replace('/','$ç')+'.joblib'
    if os.path.exists(save_file_in):
        try:
            return joblib.load(save_file_in)[-1]
        except (EOFError,KeyError):
            pass

    functions_instructions = []
    def has_numbers(inputString):
        return bool(re.search(r'\d', inputString))
    
    def is_hex(s):
        try:
            int(s, 16)
            return True
        except ValueError:
            return False

    def token_cleaner(token:str):
        if (
            '$' in token 
            or "@" in token 
            or '=' in token
            or 'nan' == token
        ):
            return ''
        token = token.strip()
        token, *args = token.split(' ')
        *types, token = token.split('.')
        i = 1
        while len(types)>=i:
            if types[-i] in {'local', 'global', 'atomic', 'memory'}:
                token = f"{types[-i]}.{token}"
                i+=1
            else:
                break
        cleaner = lambda x, w : x.replace(f'{w}x', w)
        token = cleaner(token, 'load')
        token = cleaner(token, 'store')
        token = cleaner(token, 'convert')
        token = token.replace('8', '').replace('16', '').replace('32', '').replace('64', '')
        token = re.sub(r'_i(?![a-zA-Z])', r'', token)
        token = re.sub(r'_s(?![a-zA-Z])', r'', token)
        token = re.sub(r'_u(?![a-zA-Z])', r'', token)
        token = re.sub(r'_f(?![a-zA-Z])', r'', token)
        token = re.sub(r'_v(?![a-zA-Z])', r'', token)
        token = re.sub(r'_x(?![a-zA-Z])', r'', token)
        token = token.replace('-','')
        token = token.replace('+','')
        if (
            has_numbers(token) 
            or is_hex(token.replace('p',''))
            or len(token.replace('p',''))==0
        ):
            return ''
        return token

    # def run(args):
    #     file,func = args
    #     r = subprocess.run([
    #         'wassail', 'function-body', file, str(func),
    #     ], capture_output=True).stdout.decode().split('\n')
    #     lst = [token_cleaner(x) for x in r if len(x)>0]
    #     return [lst[i] for i in range(len(lst)) if i == 0 or lst[i] != lst[i-1]]

    if print_logs:
        print("Parsing Module")
    result, _ = parse_wasm_module(file, print_logs=print_logs)
    module = result[1:]
    
    def run(args):
        part = args
        r = [str(x) for x in part[2:] if ( not isinstance(x, list)) ]
        lst = [token_cleaner(x) for x in r if len(x)>0]
        lst = [x for x in lst if len(x)>0]
        return [lst[i] for i in range(len(lst)) if i == 0 or lst[i] != lst[i-1]]
    
    with ThreadPoolExecutor(max_workers=15) as pool:
        results = list((pool.map(run, [(func) for func in [part for part in module if str(part[0]) == 'func']])))
    
    for tokens in results:
        functions_instructions.append(tokens)

    to_return = functions_instructions
    # joblib.dump(to_return,save_file_in)
    return to_return

def preprocess_image_for_training(view, device, preprocessor=None):
    if preprocessor is not None:
        view =[preprocessor(f) for f in view]
    return torch.tensor(np.array(view, dtype=np.uint8), device=device, dtype=torch.float32)

def get_joblib_or_none(f):
    try:
        return joblib.load(f)
    except Exception as e:
        print(e)
        return None

class graph_prerocessor_cls:
    def __init__(self, seq_size=512):
        try:
            self.vocab = joblib.load('metadata/vocab')
        except:
            processed_files = [
                "/Users/walidthekraken/Documents/uqam/WALISM/datasets/wasm-shield-graphs-2/"+x
                for x in
                list(os.walk("/Users/walidthekraken/Documents/uqam/WALISM/datasets/wasm-shield-graphs-2/"))[0][2]
            ]
            print('Got files')
            programs = []
            for f in tqdm.tqdm(
                random.sample(list(processed_files), k=5000)
            ):
                if prog:=get_joblib_or_none(f):
                    programs.append(prog)
            vocab_set = self.get_vocab_set_from_list_of_wasms(programs)
            # with ThreadPoolExecutor(max_workers=10) as pool:
            #     vocab_set = self.get_vocab_set_from_list_of_wasms(
            #         [
            #             x for x in
            #             list(
            #                 tqdm.tqdm([
            #                         get_joblib_or_none(f) for f in processed_files
            #                     ],
            #                     total=len(processed_files)
            #                 )
            #             )
            #             if x is not None
            #         ]
            #     )
            self.vocab = {v:i for i,v in enumerate(vocab_set)}
            joblib.dump(
                self.vocab,
                'metadata/vocab'
            )
            print(self.vocab)
        self.vocab_size = len(self.vocab.keys()) + 4
        self.unk_token = self.vocab_size - 1
        self.pad_token = self.vocab_size - 2
        self.mask_token = self.vocab_size - 3
        self.cls_token = self.vocab_size - 4
        self.seq_size = seq_size

    def get_vocab_set_from_list_of_wasms(
            self, 
            programs
        ):
        vocab_set = set()
        for program in programs:
            _, _, functions_instructions = program
            for function_instructions in functions_instructions:
                for inst in function_instructions:
                    vocab_set.add(inst)
        return vocab_set

    def get_token_ids(self, tokens, vocab,):
        trimmed_tokens = tokens[0:self.seq_size-1]
        return (
            [self.cls_token] + 
            [vocab.get(token, self.unk_token) for token in trimmed_tokens] + 
            [self.pad_token for _ in range(self.seq_size-1-len(trimmed_tokens))]
        )

    def preprocess_graph_for_training(self, view, device):
        data = []
        for adj_lists, edge_types, functions_instructions in view:
            nodes, edge_list = self.graph_preprocessor(adj_lists, edge_types, functions_instructions)
            data.append(
                (
                    torch.tensor(nodes, dtype=torch.float32, device=device), 
                    torch.tensor(edge_list, dtype=torch.long, device=device)
                )
            ) 
        return self.get_batched_graph(data)
    
    def preprocess_text_for_training(self, view, device, max_nb_func=100, min_func_length=0):
        data = []
        programs = [
            self.text_preprocessor(program, min_func_length=min_func_length, max_nb_func=max_nb_func)
            for program in view
        ]

        nb_to_pad = max([x.shape[0] for x in programs])

        batch = np.array(
            [
                np.concatenate(
                    [
                        program, 
                        self.pad_token*np.ones((nb_to_pad-program.shape[0],self.seq_size))
                    ]
                )
                
            for program in programs]
        )

        return torch.tensor(batch, dtype=torch.long, device=device)
    
    def graph_preprocessor(self, adj_lists, edge_types, functions_instructions, min_func_length=0, max_nb_func=-1):
        words_ids = [ 
            self.get_token_ids(t, self.vocab) for t in (sorted(functions_instructions, key=lambda x:len(x), reverse=True) if len(functions_instructions)>0 else [[]] )
            # if len(t)>min_func_length
        ][:max_nb_func]
        return np.array(words_ids), adj_lists
    
    def text_preprocessor(self, functions_instructions, min_func_length=0, max_nb_func=-1):
        words_ids = [ 
            self.get_token_ids(t, self.vocab) for t in (sorted(functions_instructions, key=lambda x:len(x), reverse=True) if len(functions_instructions)>0 else [[]] )
            # if len(t)>min_func_length
        ][:max_nb_func]
        return np.array(words_ids)
    
    def get_batched_graph(self, data):
        n_nodes = 0
        nodes_tensors = []
        edge_tensors = []
        n_nodes_per_graph = []
        for i, (nodes_tensor, edge_list_tensor) in enumerate(data):
            nodes_tensors.append(torch.add(nodes_tensor, n_nodes))
            edge_tensors.append(torch.add(edge_list_tensor, n_nodes))
            n = nodes_tensor.shape[0]
            if n==0:
                print(nodes_tensor)
            n_nodes_per_graph.extend([i]*n)
            n_nodes+=(n+1)
        return torch.concatenate(nodes_tensors, dim=0), torch.concatenate(edge_tensors, dim=1), np.array(n_nodes_per_graph)

