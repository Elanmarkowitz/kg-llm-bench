from llm import openai, gemini

class LLM:
    def __init__(self, model='gpt-4o-mini', provider='openai') -> None:
        self.model = model
        self.provider = provider

    def __call__(self, prompt, max_tokens=150, temperature=0):
        if self.provider == 'openai':
            return openai.llm(self.model, prompt, max_tokens, temperature)
        elif self.provider == 'google':
            return gemini.llm(self.model, prompt, max_tokens, temperature)




        