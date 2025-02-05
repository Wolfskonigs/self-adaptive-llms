import os

from .base import BaseModel


class Llama3Instruct8B(BaseModel):
    def __init__(self):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.dec_param_file_n = "llama3_decomposed_params.pt"

    def get_model_id(self):
        return self.model_id

    def get_model_name(self):
        return self.model_id.split("/")[1]

    def get_param_file(self, param_folder_path=""):
        return os.path.join(param_folder_path, self.dec_param_file_n)
