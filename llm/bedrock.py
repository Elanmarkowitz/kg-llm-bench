import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
import argparse
import os
from botocore.exceptions import ClientError

def init_bedrock_client(region=None):
    """
    Initialize the Bedrock client with proper credentials and region.
    
    Args:
        region (str): AWS region. If None, will try to get from environment or default to us-east-1
    """
    # Check for AWS credentials
    if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.\n"
            "You can do this by:\n"
            "1. Export them in your shell:\n"
            "   export AWS_ACCESS_KEY_ID=your_access_key\n"
            "   export AWS_SECRET_ACCESS_KEY=your_secret_key\n"
            "2. Or configure AWS CLI: aws configure"
        )
    
    # Get region from environment or use default
    region = region or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1'
    
    try:
        client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region
        )
        # Test the client with a simple API call
        client.list_tags_for_resource(resourceARN='arn:aws:bedrock:' + region)
        return client
    except ClientError as e:
        if 'AccessDeniedException' in str(e):
            raise ValueError(
                "Access denied. Please ensure your AWS credentials have permission to access Bedrock.\n"
                "You may need to:\n"
                "1. Enable Bedrock in your AWS account\n"
                "2. Grant necessary IAM permissions\n"
                "3. Request model access in the Bedrock console"
            )
        elif 'UnrecognizedClientException' in str(e):
            raise ValueError("Invalid AWS credentials. Please check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        else:
            raise e

def llm(model_name, prompt, max_tokens=150, temperature=0):
    """
    Calls the AWS Bedrock API with the specified model and prompt using LangChain.

    Parameters:
    - model_name (str): The name of the model to use. Supported models:
        - Anthropic: 'anthropic.claude-v2', 'anthropic.claude-v2:1', 'anthropic.claude-3-sonnet-20240229-v1:0'
        - Meta: 'meta.llama2-13b-chat-v1', 'meta.llama2-70b-chat-v1'
        - Mistral: 'mistral.mixtral-8x7b-instruct-v1'
        - Cohere: 'cohere.command-text-v14', 'cohere.command-light-text-v14'
        - AI21: 'ai21.j2-ultra-v1', 'ai21.j2-mid-v1'
        - Amazon: 'amazon.titan-text-express-v1', 'amazon.titan-text-lite-v1'
    - prompt (str): The prompt to send to the model.
    - max_tokens (int): Maximum number of tokens to generate.
    - temperature (float): Controls randomness in the response (0-1).

    Returns:
    - tuple: (response_text, usage_stats)
    """
    try:
        # Initialize Bedrock client with proper configuration
        bedrock_client = init_bedrock_client()
        
        # Initialize the LangChain AWS Bedrock chat model
        chat = ChatBedrock(
            model_id=model_name,
            client=bedrock_client,
            model_kwargs={
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        
        # Create message and get response
        message = HumanMessage(content=prompt)
        response = chat([message])
        
        # Extract completion and usage stats
        completion = response.content
        
        # Get token usage if available
        usage = {}
        if hasattr(response, 'additional_kwargs') and 'usage' in response.additional_kwargs:
            usage = response.additional_kwargs['usage']
        else:
            usage = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
            
        return completion.strip(), usage
        
    except Exception as e:
        raise Exception(f"Error calling Bedrock API via LangChain: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test AWS Bedrock models')
    parser.add_argument('--model', type=str, help='Model to use (e.g., anthropic.claude-v2)')
    parser.add_argument('--prompt', type=str, help='Prompt to send to the model')
    parser.add_argument('--max-tokens', type=int, default=150, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature (0-1)')
    parser.add_argument('--region', type=str, help='AWS region (default: us-east-1)')
    
    args = parser.parse_args()
    
    if not args.model or not args.prompt:
        parser.print_help()
        exit(1)
    
    try:
        print(f"\nSending prompt to {args.model}...")
        print(f"Prompt: {args.prompt}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Temperature: {args.temperature}")
        print(f"Region: {args.region or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1'}")
        print("\nGenerating response...")
        
        response, usage = llm(
            model_name=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print("\nResponse:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        print("\nToken Usage:")
        print(f"Prompt tokens: {usage.get('prompt_tokens', 0)}")
        print(f"Completion tokens: {usage.get('completion_tokens', 0)}")
        print(f"Total tokens: {usage.get('total_tokens', 0)}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")


