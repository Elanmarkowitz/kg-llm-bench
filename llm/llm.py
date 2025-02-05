from llm import openai, gemini, bedrock, batch_bedrock
import os

class LLM:
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
            if not hasattr(self, '_batch_provider'):
                self._batch_provider = batch_bedrock.BatchBedrock(
                    model=self.model,
                    batch_dir='batch_data',
                    max_batch_size=int(os.getenv('BATCH_SIZE', '100')),
                    batch_timeout_minutes=int(os.getenv('BATCH_TIMEOUT', '60')),
                    s3_input_bucket=os.getenv('BEDROCK_INPUT_BUCKET'),
                    s3_output_bucket=os.getenv('BEDROCK_OUTPUT_BUCKET')
                )
            return self._batch_provider(prompt, max_tokens, temperature)



        