from llm import openai

class LLM:
    def __init__(self, llm_config) -> None:
        self.llm_config = llm_config

    def __call__(self, prompt, max_tokens=150, temperature=0):
        if self.llm_config['provider'] == 'openai':
            return openai.llm(self.llm_config['model'], prompt, max_tokens, temperature)
        