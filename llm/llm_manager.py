from langchain.llms import HuggingFacePipeline
from utils.config_loader import load_config
from transformers import pipeline

def load_llm(config_path="config/app_config.yaml"):
    config = load_config(config_path)
    llm_type = config['llm_model']['type']
    llm_path = config['llm_model']['path']

    if llm_type == "llama":
        generator = pipeline("text-generation", model=llm_path)
        return HuggingFacePipeline(pipeline=generator)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
