from .base import BaseModel
from typing import List

class VLLMModel(BaseModel):
    '''
    Basic infrastructure for using VLLM models
    Most arguments should default to the VLLM defaults but this provides nice flexibility
    '''
    def __init__(self, model_name_or_path: str, model_args : dict = {}, generation_args: dict = ..., batch_size: int = 1) -> None:
        super().__init__(model_name_or_path, None, generation_args, batch_size)
        from vllm import LLM, SamplingParams, EngineArgs, LLMEngine

        args = EngineArgs(model=model_name_or_path, **model_args)
        engine = LLMEngine.from_engine_args(args)
    
        self.generation_args = SamplingParams(**generation_args)
        self.model = LLM(engine, self.generation_args)

    def generate(self, inps : List[str]) -> List[str]:
        outputs = self.model.generate(inps, self.generation_args)

        return [output[0].text for output in outputs]