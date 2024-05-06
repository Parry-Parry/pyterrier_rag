from .base import BaseModel

class VLLMModel(BaseModel):
    def __init__(self, model_name_or_path: str, config: Any = None, generation_args: dict = ..., batch_size: int = 1) -> None:
        super().__init__(model_name_or_path, config, generation_args, batch_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None

