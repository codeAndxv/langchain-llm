import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from configs.model_config import LLM_DEVICE


class LoaderCheckPoint:
    """
    加载自定义 model CheckPoint
    """
    # 模型名称
    model_name: str = None
    tokenizer: object = None
    # 模型全路径
    model: object = None
    model_config: object = None
    lora_names: set = []
    lora_dir: str = None
    ptuning_dir: str = None
    use_ptuning_v2: bool = False

    is_llamacpp: bool = False
    params: object = None

    def __init__(self, params: dict = None):
        """
        模型初始化
        :param params:
        """
        self.model = None
        self.tokenizer = None
        self.params = params or {}
        self.no_remote_model = params.get('no_remote_model', False)

