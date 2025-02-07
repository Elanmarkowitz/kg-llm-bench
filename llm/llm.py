from llm import openai, gemini, bedrock
from llm.batch_manager import BatchManager
import os

class LLM:
    # Shared batch manager instance
    _batch_manager = BatchManager()
    
    def __init__(self, model='gpt-4o-mini', provider='openai') -> None:
        self.model = model
        self.provider = provider

    def __call__(self, prompt, max_tokens=512, temperature=0):
        if self.provider == 'openai':
            return openai.llm(self.model, prompt, max_tokens, temperature)
        elif self.provider == 'google':
            return gemini.llm(self.model, prompt, max_tokens, temperature)
        elif self.provider == 'bedrock':
            return bedrock.llm(self.model, prompt, max_tokens, temperature)
        elif self.provider == 'batch_bedrock':
            # Get shared batch provider for this model
            batch_provider = self._batch_manager.get_provider(self.model)
            return batch_provider(prompt, max_tokens, temperature)



        