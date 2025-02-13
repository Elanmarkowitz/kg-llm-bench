#!/usr/bin/env python3
import io
import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from io import BytesIO

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langchain_aws.llms.bedrock import LLMInputOutputAdapter

load_dotenv()

def init_clients(region: Optional[str] = None):
    """Initialize AWS clients."""
    region = region or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1'
    bedrock = boto3.client('bedrock', region_name=region)
    s3 = boto3.client('s3')
    return bedrock, s3

def download_results(s3_client, uri: str, local_path: Path, metadata: Dict):
    """Download results from S3."""
    bucket = uri.split('/')[2]
    base_key = '/'.join(uri.split('/')[3:])
    base_key = base_key.rstrip('/')  # Remove trailing slash if present
    
    try:
        # Extract job ID from job ARN (last part after final /)
        dir_key = f"{base_key}/{metadata['job_arn'].split('/')[-1]}"
            
        # Download manifest file
        manifest_key = f"{dir_key}/manifest.json.out"
        manifest_path = local_path.with_name('manifest.json.out')
        s3_client.download_file(bucket, manifest_key, str(manifest_path))
        
        # Download records file
        records_key = f"{dir_key}/records.jsonl.out"
        records_path = local_path.with_name('records.jsonl.out')
        s3_client.download_file(bucket, records_key, str(records_path))
        
        return True
            
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        return False

def process_batch_results(results_file: Path, metadata: Dict) -> Dict[str, Dict]:
    """Process batch results and map them back to record IDs."""
    processed_results = {}
    has_errors = False
    
    # Get provider from model ID
    model_id = metadata['model']
    provider = model_id.split('.')[0]
    if provider not in ['anthropic', 'meta', 'mistral', 'amazon']:
        provider = model_id.split('.')[1]
    if provider not in ['anthropic', 'meta', 'mistral', 'amazon']:
        raise ValueError(f"Unable to identify Bedrock provider for model: {model_id}")
    
    # Process records file
    records_file = results_file.with_name('records.jsonl.out')
    if records_file.exists():
        with open(records_file, 'r') as f:
            for line in f:
                result = json.loads(line)
                record_id = result['recordId']
                
                # Check if there's an error in the result
                if 'error' in result:
                    has_errors = True
                    processed_results[record_id] = {
                        'completion': '',
                        'usage': {},
                        'status': 'failed',
                        'error': result.get('error', 'Unknown error')
                    }
                    continue

                # Use LLMInputOutputAdapter to process the output
                try:
                    # Create a readable bytes object
                    body_bytes = io.BytesIO(json.dumps(result['modelOutput']).encode())
                    
                    output = LLMInputOutputAdapter.prepare_output(provider, {
                        'body': body_bytes,
                        'ResponseMetadata': {
                            'HTTPHeaders': {
                                'x-amzn-bedrock-input-token-count': result['modelOutput']['usage']['input_tokens'],
                                'x-amzn-bedrock-output-token-count': result['modelOutput']['usage']['output_tokens']
                            }
                        }
                    })
                    
                    processed_results[record_id] = {
                        'completion': output['text'],
                        'usage': output['usage'],
                        'status': 'completed',
                        'tool_calls': output['tool_calls']
                    }
                except Exception as e:
                    print(f"Error processing result for record {record_id}: {e}")
                    processed_results[record_id] = {
                        'completion': '',
                        'usage': {},
                        'status': 'failed',
                        'error': str(e)
                    }
                    has_errors = True
    
    return processed_results, has_errors

def update_task_results(results_dir: Path, record_results: Dict[str, Dict]):
    """Update task result files with batch results."""
    # Scan all result files in the results directory
    for results_file in results_dir.rglob('*_results.json'):
        updated = False
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        for instance in results:
            if instance is None:
                continue
                
            response = instance.get('response', '')
            if isinstance(response, str) and response.startswith('PENDING:'):
                record_id = response.split(':')[1]
                if record_id in record_results:
                    result = record_results[record_id]
                    instance['response'] = result['completion']
                    instance['usage_tokens'] = result['usage']
                    instance['tool_calls'] = result.get('tool_calls', [])
                    updated = True
        
        if updated:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Updated results file: {results_file}")

def process_submitted_batch(bedrock_client, s3_client, batch_path: Path):
    """Process a submitted batch and collect results if complete."""
    with open(batch_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    if 'job_arn' not in metadata:
        print(f"No job ARN found for batch: {batch_path}")
        return
    
    try:
        job = bedrock_client.get_model_invocation_job(
            jobIdentifier=metadata['job_arn']
        )
        job_status = job['status']
        
        if job_status == 'Completed':
            print(f"Processing completed batch: {batch_path}")
            
            # Download results - create a dummy results file path just for the download function
            results_file = batch_path / 'results.jsonl'
            if download_results(s3_client, metadata['s3_output_uri'], results_file, metadata):
                # Process results using the actual manifest and records files
                results, has_errors = process_batch_results(results_file, metadata)
                if has_errors:
                    print(f"Errors found in batch results: {batch_path}")
                    metadata['status'] = 'failed'
                    metadata['error'] = "Errors found in batch results"
                    with open(batch_path / 'metadata.json', 'w') as f:
                        json.dump(metadata, f, indent=2)
                    return
                
                # Update task results
                update_task_results(Path('benchmark_data'), results)
                
                # Move to completed
                completed_path = Path('batch_data/completed') / batch_path.name
                batch_path.rename(completed_path)
                
                print(f"Successfully processed batch: {batch_path}")
            
        elif job_status in ['Failed', 'Stopped']:
            print(f"Batch job failed or stopped: {batch_path}")
            metadata['status'] = 'failed'
            metadata['error'] = f"Job status: {job_status}"
            with open(batch_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        else:
            print(f"Batch job still in progress: {batch_path} (status: {job_status})")
            
    except ClientError as e:
        print(f"Error checking job status: {e}")

def main():
    parser = argparse.ArgumentParser(description='Collect and process completed Bedrock batch results')
    parser.add_argument('--region', help='AWS region (default: from environment)')
    parser.add_argument('--batch-dir', default='batch_data', help='Base directory for batch data')
    args = parser.parse_args()
    
    # Initialize clients
    bedrock_client, s3_client = init_clients(args.region)
    
    # Process all submitted batches
    submitted_dir = Path(args.batch_dir) / 'submitted'
    if not submitted_dir.exists():
        print(f"No submitted directory found at {submitted_dir}")
        return
    
    for batch_path in submitted_dir.iterdir():
        if batch_path.is_dir():
            process_submitted_batch(bedrock_client, s3_client, batch_path)
            

if __name__ == "__main__":
    main() 