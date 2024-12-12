from langchain.llms import HuggingFacePipeline
from utils.config_loader import load_config
from transformers import pipeline


from transformers import pipeline
from langchain.llms import HuggingFacePipeline

def load_llm(config_path="config/llm_config.yaml"):
    config = load_config(config_path)
    llm_type = config['llm_type']
    model_path = config['model_path']
    parameters = config.get('parameters', {})

    if llm_type == "llama":
        generator = pipeline("text-generation", model=model_path, **parameters)
        return HuggingFacePipeline(pipeline=generator)
    elif llm_type == "flan-t5":
        generator = pipeline("text2text-generation", model=model_path, **parameters)
        return HuggingFacePipeline(pipeline=generator)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

