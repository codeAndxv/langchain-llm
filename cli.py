from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
import models.shared as shared
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Show reply with source text from input document
REPLY_WITH_SOURCE = False


def main():
    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    filepath = "D:\\qa"
    vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath)

    llm_model_ins = shared.loaderLLM(LLM_MODEL)
    llm_model_ins.history_len = LLM_HISTORY_LEN

    local_doc_qa.llm = llm_model_ins

    history = []
    query = "模型量化"
    for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                 vs_path=vs_path,
                                                                 chat_history=history,
                                                                 streaming=STREAMING):
        print(resp["result"])


if __name__ == "__main__":
    main()
