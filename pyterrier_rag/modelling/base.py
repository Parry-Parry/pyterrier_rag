from typing import Any, Union
from pandas import DataFrame
import pyterrier as pt
if not pt.started():
    pt.init()
from abc import abstractmethod, ABC

class BaseModel(object, ABC):
    '''
    Simple wrapper to encourage this format, but not necessary anything which can take in a list of strings can be used

    Custom behaviour would be needed to allow multiple string outputs for a single input within BaseReader
    '''
    def __init__(self, 
                 model_name_or_path : str, 
                 config : Any = None,
                 generation_args : dict = {},
                 batch_size : int = 1,
                 ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.config = config if config is not None else None
        self.generation_args = generation_args
        self.batch_size = batch_size
        self.max_input_length = self.config.max_position_embeddings
        self.max_output_length = self.config.max_position_embeddings
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        raise NotImplementedError("generate method must be implemented")

class BaseReader(pt.Transformer, ABC):
    def __init__(self, 
                 model : Union[BaseModel, Any], 
                 query_field : str = 'query', 
                 text_field : str = 'text', 
                 output_field : str = 'answer',
                 ) -> None:
        super().__init__()
        self.model = model
        self.query_field = query_field
        self.text_field = text_field
        self.output_field = output_field
    
    @abstractmethod
    def construct_context(self, *args, **kwargs):
        '''
        Is abstracted to allow context processing beyond topics_or_res.groupby(['qid', self.query_field])[self.text_field].apply(list)
        '''
        raise NotImplementedError("construct_context method must be implemented")
    
    @abstractmethod
    def construct_prompt(self, *args, **kwargs):
        '''
        Should take in a single example set and return a prompt string, multiprocessing can be implemented by the user
        '''
        raise NotImplementedError("construct_prompt method must be implemented")
    
    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        '''
        Assumes that the input DataFrame contains the context items for each query
        '''
        output = {
            'qid' : [],
            'query' : [], 
            'context' : [],
            'prompt' : [],
            self.output_field : [],
        }
        for (qid, query), documents in topics_or_res.groupby(['qid', self.query_field]):
            output['qid'].append(qid)
            output['query'].append(query)
            context = self.construct_context(documents)
            output['context'].append(context)
            output['prompt'] = self.construct_prompt(query, context)

        prompts = output['prompt'].tolist()
        output[self.output_field] = self.model.generate(prompts)

        return DataFrame(output)


        

