import torch
from typing import Any, List
from .base import BaseModel

class HFModel(BaseModel):
    def __init__(self, model_name_or_path: str, config: Any = None, generation_args: dict = ..., batch_size: int = 1) -> None:
        super().__init__(model_name_or_path, config, generation_args, batch_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
    
    def generate(self, inps : List[str]) -> List[str]:
        assert self.model is not None, "Model is not loaded, you should instantiate a subclass of HFModel"
        inputs = self.tokenizer(inps, return_tensors='pt', padding=True, truncation=True, max_length=self.max_input_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, **self.generation_args)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

class CausalModel(HFModel):
    def __init__(self, 
                 model_name_or_path: str, 
                 model_args : dict = {},
                 config: Any = None, 
                 generation_args: dict = ..., 
                 batch_size: int = 1
                 ) -> None:
        super().__init__(model_name_or_path, config, generation_args, batch_size)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_args).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

class Seq2SeqModel(HFModel):
    def __init__(self, 
                 model_name_or_path: str, 
                 model_args : dict = {},
                 config: Any = None, 
                 generation_args: dict = ..., 
                 batch_size: int = 1
                 ) -> None:
        super().__init__(model_name_or_path, config, generation_args, batch_size)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_args).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)