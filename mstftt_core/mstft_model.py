from langchain_openai import ChatOpenAI
from mstftt_core.mstft_config import get_client_config

###############################################################################
# Model
###############################################################################


class Model:
    def __init__(self, name: str, model_checkpoint: str, temperature=1, top_p=1, presence_penalty=1, frequency_penalty=0) -> None:
        self.model = ChatOpenAI(
            model=model_checkpoint,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )
        self.name = name
        self.temperature = temperature
        self.model_checkpoint = model_checkpoint
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.info = f"model name: {name}\nmodel checkpoint: {model_checkpoint}\ntemperature: {temperature}\ntop p: {top_p}\npresence penalty: {presence_penalty}\nfrequency penalty: {frequency_penalty}"


# Initialize the LLM model for prompting
llm = Model('taxonomy construction', 'gpt-4o', temperature=0.2, top_p=0.90,
                presence_penalty=0.80, frequency_penalty=0.30)
#global model 
model = llm.model