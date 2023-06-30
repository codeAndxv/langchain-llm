import gc
import time
from pathlib import Path
import torch
import transformers
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, LlamaTokenizer)
from configs.model_config import LLM_DEVICE

class LoaderCheckPoint:
    """
    加载自定义 model CheckPoint
    """
    # 模型名称
    model_name: str = None
    tokenizer: object = None
    # 模型全路径
    model_path: str = None
    model: object = None
    model_config: object = None

    params: object = None
    # 默认 cuda ，如果不支持cuda使用多卡， 如果不支持多卡 使用cpu
    llm_device = LLM_DEVICE
    def __init__(self, params: dict = None):
        """
        模型初始化
        :param params:
        """
        self.model = None
        self.tokenizer = None
        self.params = params or {}

    def load_model(self, model_name):
        """
        加载自定义位置的model
        :param model_name:
        :return:
        """
        print(f"Loading {model_name}...")
        t0 = time.time()
        # Load the model in simple 16-bit mode by default
        # 如果加载没问题，但在推理时报错RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
        # 那还是因为显存不够，此时只能考虑--load-in-8bit,或者配置默认模型为`chatglm-6b-int8`
        print(
            "Warning: self.llm_device is False.\nThis means that no use GPU  bring to be load CPU mode\n")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, revision="v1.0").float()

        print(f"Loaded the model in {(time.time() - t0):.2f} seconds.")
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def clear_torch_cache(self):
        gc.collect()
        if self.llm_device.lower() != "cpu":
            if torch.has_mps:
                try:
                    from torch.mps import empty_cache
                    empty_cache()
                except Exception as e:
                    print(e)
                    print(
                        "如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")
            elif torch.has_cuda:
                device_id = "0" if torch.cuda.is_available() else None
                CUDA_DEVICE = f"{self.llm_device}:{device_id}" if device_id else self.llm_device
                with torch.cuda.device(CUDA_DEVICE):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            else:
                print("未检测到 cuda 或 mps，暂不支持清理显存")

    def unload_model(self):
        del self.model
        del self.tokenizer
        self.model = self.tokenizer = None
        self.clear_torch_cache()

    def reload_model(self):
        self.unload_model()
        self.model, self.tokenizer = self.load_model(self.model_name)
