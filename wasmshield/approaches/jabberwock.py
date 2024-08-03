import os
import numpy as np
import subprocess

from wasmshield.models.base_handler import BaseHandler
from concurrent.futures import ThreadPoolExecutor

class JabberWockHandler(BaseHandler):

    def __init__(self, model, save_vectors_in):
        self.model = model
        self.save_vectors_in = save_vectors_in

    def preprocess_file(self, file:str, size:int=0):
        f = open(file,'rb')
        bytes = f.read()
        f.close()
        return self.preprocess_bytes(bytes, size)
    
    def preprocess_bytes(self, bytes:bytes, size:int=0):
        p = subprocess.run([ '/opt/homebrew/bin/wasm-tools' , 'print' ,'-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=bytes)
        watdata = p.stdout.decode()
        watsplit = watdata.split("\n")
        return watsplit
    
    def get_vector_from_preprocessed_file(self, preprocessed_file, device='mps'):
        vec = self.model.infer_vector(preprocessed_file)
        return vec
    
    def get_vector_from_file(self, filename):
        vec_filename = self.save_vectors_in+filename.replace('/','$ç')+'.npy'
        if os.path.exists(vec_filename):
            try:
                return np.load(vec_filename)
            except EOFError:
                pass
        watsplit = self.preprocess_file(filename)
        vec = self.model.infer_vector(watsplit)
        np.save(vec_filename, vec)
        return vec
    
    def get_vectors_from_files(
        self, 
        files, 
        size:int=0, 
        max_workers=24,
    ):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            vectors = [v for v in (list(pool.map(self.get_vector_from_file, files)))]
        vectors = np.stack(list(vectors), axis=0)
        return vectors
