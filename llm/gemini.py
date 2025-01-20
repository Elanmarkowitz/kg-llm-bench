import google.generativeai as genai
import os


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)

if GEMINI_API_KEY is not None:
    genai.configure(api_key=GEMINI_API_KEY)


def llm(model_name, prompt, max_tokens=150, temperature=0):
    """
    Calls the Gemini API with the specified model and prompt.

    Parameters:
    - model_name (str): The name of the model to use.
    - prompt (str): The prompt to send to the model.
    - max_tokens (int): The maximum number of tokens to generate.
    - temperature (float): The sampling temperature.

    Returns:
    - response (str): The response from the Gemini API.
    """

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )
    response = model.generate_content(prompt, generation_config=generation_config)
    usage_token = {
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count,
        "total_tokens": response.usage_metadata.total_token_count,
        'cached_content_token': response.usage_metadata.cached_content_token_count
    }
    return response.text, usage_token