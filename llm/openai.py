from openai import OpenAI
client = OpenAI()

def llm(model_name, prompt, max_tokens=150, temperature=0):
    """
    Calls the OpenAI API with the specified model and prompt.

    Parameters:
    - model_name (str): The name of the model to use.
    - prompt (str): The prompt to send to the model.
    - kwargs: Additional parameters for the OpenAI API call (e.g., temperature, max_tokens).

    Returns:
    - response (dict): The response from the OpenAI API.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content, dict(response.usage)