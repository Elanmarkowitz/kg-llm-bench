import os
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from langchain_aws.chat_models.bedrock import ChatPromptAdapter
from langchain_aws.llms.bedrock import LLMInputOutputAdapter
from langchain_core.messages import HumanMessage, SystemMessage

class BatchBedrock:
    def __init__(self, 
                 model: str,
                 batch_dir: str = 'batch_data',
                 max_batch_size: int = 100,
                 batch_timeout_minutes: int = 60,
                 s3_input_bucket: Optional[str] = None,
                 s3_output_bucket: Optional[str] = None) -> None:
        """
        Initialize BatchBedrock for accumulating and processing batch requests.
        
        Args:
            model: The Bedrock model identifier
            batch_dir: Directory to store batch data
            max_batch_size: Maximum number of records per batch
            batch_timeout_minutes: Time before starting new batch
            s3_input_bucket: S3 bucket for input data
            s3_output_bucket: S3 bucket for output data
        """
        self.model = model
        self.batch_dir = Path(batch_dir)
        self.max_batch_size = max_batch_size
        self.batch_timeout_minutes = batch_timeout_minutes
        self.s3_input_bucket = s3_input_bucket
        self.s3_output_bucket = s3_output_bucket
        
        # Create directory structure
        self.pending_dir = self.batch_dir / 'pending'
        self.submitted_dir = self.batch_dir / 'submitted'
        self.completed_dir = self.batch_dir / 'completed'
        
        for dir in [self.pending_dir, self.submitted_dir, self.completed_dir]:
            dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize current batch
        self.current_batch = None
        self.current_batch_path = None
        self.current_batch_size = 0
        self.current_batch_start = None
        
    def __call__(self, prompt: str, max_tokens: int = 512, temperature: float = 0, 
                 system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Process a request by adding it to the current batch.
        
        Returns:
            Tuple[str, Dict]: (record_id, usage_stats)
        """
        # Check if we need to start a new batch
        if self._should_start_new_batch():
            self._start_new_batch()
            
        # Generate unique record ID
        record_id = f"REC{str(uuid.uuid4())[:8]}"

        # Convert prompt according to provider
        bedrock_provider = self.model.split('.')[0]
        if bedrock_provider not in ['anthropic', 'meta', 'mistral', 'amazon']:
            bedrock_provider = self.model.split('.')[1]
        if bedrock_provider not in ['anthropic', 'meta', 'mistral', 'amazon']:
            raise ValueError(f"Unable to identify Bedrock provider for model: {self.model}")

        # Convert messages to provider-specific prompt format
        messages = [HumanMessage(content=prompt)]
        if system_prompt:
            messages.insert(0, SystemMessage(content=system_prompt))

        try:
            messages = ChatPromptAdapter.format_messages(bedrock_provider, messages)
            prompt = None
        except NotImplementedError:
            messages = None
            prompt = ChatPromptAdapter.convert_messages_to_prompt(bedrock_provider, messages, self.model) 
            
        # Create model input using LLMInputOutputAdapter
        model_input = LLMInputOutputAdapter.prepare_input(
            provider=bedrock_provider,
            prompt=prompt,
            system_prompt=None,
            messages=messages,
            model_kwargs={},
            max_tokens=max_tokens,
            temperature=temperature
        )
        # Create record
        record = {
            "recordId": record_id,
            "modelInput": model_input
        }
        
        # Append to current batch
        with open(self.current_batch_path / 'records.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')
            
        # Update metadata
        self._update_metadata(record_id, prompt, system_prompt)
        
        self.current_batch_size += 1
        
        # Return pending response
        usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'status': 'pending',
            'record_id': record_id
        }
        
        return f"PENDING:{record_id}", usage
    
    def _should_start_new_batch(self) -> bool:
        """Check if we should start a new batch based on size and time."""
        if self.current_batch is None:
            return True
            
        if self.current_batch_size >= self.max_batch_size:
            return True
            
        if self.current_batch_start:
            elapsed_minutes = (datetime.now() - self.current_batch_start).total_seconds() / 60
            if elapsed_minutes >= self.batch_timeout_minutes:
                return True
                
        return False
    
    def _start_new_batch(self):
        """Start a new batch by creating necessary directories and files."""
        batch_id = f"batch-{int(time.time())}-{str(uuid.uuid4())[:8]}"
        batch_path = self.pending_dir / batch_id
        batch_path.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'model': self.model,
            'created_at': datetime.now().isoformat(),
            'status': 'pending',
            'record_map': {},
            'record_count': 0
        }
        
        with open(batch_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Create empty records file
        (batch_path / 'records.jsonl').touch()
        
        self.current_batch = batch_id
        self.current_batch_path = batch_path
        self.current_batch_size = 0
        self.current_batch_start = datetime.now()
    
    def _update_metadata(self, record_id: str, prompt: str, system_prompt: Optional[str]):
        """Update batch metadata with new record information."""
        metadata_path = self.current_batch_path / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        metadata['record_map'][record_id] = {
            'prompt': prompt,
            'system_prompt': system_prompt,
            'timestamp': datetime.now().isoformat()
        }
        metadata['record_count'] += 1
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2) 