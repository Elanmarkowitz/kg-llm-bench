import os
import json
from pathlib import Path
from typing import Dict
from .batch_bedrock import BatchBedrock

class BatchManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BatchManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._providers: Dict[str, BatchBedrock] = {}
            self._state_file = Path('batch_data/manager_state.json')
            self._initialized = True
            self._load_state()
    
    def _load_state(self):
        """Load provider state from disk if it exists."""
        if self._state_file.exists():
            with open(self._state_file, 'r') as f:
                state = json.load(f)
                for model, provider_state in state.items():
                    if model not in self._providers:
                        self._providers[model] = BatchBedrock(
                            model=model,
                            batch_dir=provider_state.get('batch_dir', 'batch_data'),
                            max_batch_size=provider_state.get('max_batch_size', 100),
                            batch_timeout_minutes=provider_state.get('batch_timeout_minutes', 60),
                            s3_input_bucket=provider_state.get('s3_input_bucket'),
                            s3_output_bucket=provider_state.get('s3_output_bucket')
                        )
    
    def _save_state(self):
        """Save provider state to disk."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {}
        for model, provider in self._providers.items():
            state[model] = {
                'batch_dir': str(provider.batch_dir),
                'max_batch_size': provider.max_batch_size,
                'batch_timeout_minutes': provider.batch_timeout_minutes,
                's3_input_bucket': provider.s3_input_bucket,
                's3_output_bucket': provider.s3_output_bucket,
                'current_batch': provider.current_batch,
                'current_batch_size': provider.current_batch_size
            }
        with open(self._state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_provider(self, model: str) -> BatchBedrock:
        """Get or create a BatchBedrock provider for a model."""
        if model not in self._providers:
            self._providers[model] = BatchBedrock(
                model=model,
                batch_dir='batch_data',
                max_batch_size=int(os.getenv('BATCH_SIZE', '100')),
                batch_timeout_minutes=int(os.getenv('BATCH_TIMEOUT', '60')),
                s3_input_bucket=os.getenv('BEDROCK_INPUT_BUCKET'),
                s3_output_bucket=os.getenv('BEDROCK_OUTPUT_BUCKET')
            )
            self._save_state()
        return self._providers[model]
    
    def get_all_providers(self) -> Dict[str, BatchBedrock]:
        """Get all active batch providers."""
        return self._providers
    
    def reset_provider(self, model: str):
        """Reset a specific provider's state."""
        if model in self._providers:
            del self._providers[model]
            self._save_state()
    
    def reset_all(self):
        """Reset all providers."""
        self._providers.clear()
        if self._state_file.exists():
            self._state_file.unlink()
            
    def cleanup_provider(self, model: str):
        """Clean up provider resources."""
        if model in self._providers:
            provider = self._providers[model]
            if provider.current_batch:
                provider._start_new_batch()  # Close current batch
            del self._providers[model]
            self._save_state() 