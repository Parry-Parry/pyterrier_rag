
from .base import BaseReader
from .prompt import AutoPrompt
from typing import Any

class SimpleReader(BaseReader):
    def __init__(self, 
                 model : Any, 
                 query_field : str = 'query', 
                 text_field : str = 'text', 
                 output_field : str = 'answer',
                 ) -> None:
        super().__init__(model, query_field, text_field, output_field)
        self.prompt = AutoPrompt.from_string('Use the context information to answer the Question: \n Context: {context} \n Question: {query} \n Answer:')
    
    def construct_context(self, documents):
        return '\n'.join(documents[self.text_field].tolist())
    
    def construct_prompt(self, query, context):
        return self.prompt(query=query, context=context)
    
   