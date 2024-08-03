import abc

class BaseHandler():

    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def preprocess_file(
        self, 
        file:str, 
        size:int
    ):
        pass

    @abc.abstractmethod
    def preprocess_bytes(
        self, 
        bytes:bytes, 
        size:int
    ):
        pass
    
    @abc.abstractmethod
    def get_vector_from_preprocessed_file(
        self, 
        preprocessed_file, 
        device='mps'
    ):
        pass

    def get_vectors_from_files(
        self, 
        files, 
        size:int=0, 
        max_workers=24,
    ):
        pass

    