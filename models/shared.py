import sys
from typing import Any
from models.loader import LoaderCheckPoint
from configs.model_config import (llm_model_dict)

loaderCheckPoint = LoaderCheckPoint()

def loaderLLM(llm_model: str) -> Any:
    """
    init llm_model_ins LLM
    :param llm_model: model_name
    :return:
    """
    llm_model_info = llm_model_dict[llm_model]
    loaderCheckPoint.model_name = llm_model_info['name']
    loaderCheckPoint.model_path = llm_model_info["local_model_path"]

    if 'FastChatOpenAILLM' not in llm_model_info["provides"]:
        loaderCheckPoint.reload_model()

    provides_class = getattr(sys.modules['models'], llm_model_info['provides'])
    modelInsLLM = provides_class(checkPoint=loaderCheckPoint)
    if 'FastChatOpenAILLM' in llm_model_info["provides"]:
        modelInsLLM.set_api_base_url(llm_model_info['api_base_url'])
        modelInsLLM.call_model_name(llm_model_info['name'])
    return modelInsLLM

