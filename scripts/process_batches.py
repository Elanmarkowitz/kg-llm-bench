#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

def init_bedrock_client(region: Optional[str] = None):
    """Initialize the Bedrock client."""
    region = region or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1'
    return boto3.client(
        service_name='bedrock',
        region_name=region
    )

def upload_to_s3(s3_client, file_path: Path, bucket: str, key: str):
    """Upload a file to S3."""
    try:
        s3_client.upload_file(str(file_path), bucket, key)
        return f"s3://{bucket}/{key}"
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        raise

def process_pending_batch(bedrock_client, s3_client, batch_path: Path, 
                         input_bucket: str, output_bucket: str):
    """Process a single pending batch."""
    print(f"Processing batch: {batch_path}")
    
    # Load metadata
    with open(batch_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Upload records to S3
    records_key = f"batch_inputs/{batch_path.name}/records.jsonl"
    s3_input_uri = upload_to_s3(
        s3_client, 
        batch_path / 'records.jsonl',
        input_bucket,
        records_key
    )
    
    # Create batch job
    role_arn = os.getenv('BEDROCK_BATCH_ROLE_ARN')
    if role_arn is None:
        raise ValueError("The environment variable 'BEDROCK_BATCH_ROLE_ARN' is not set.")
    
    try:
        response = bedrock_client.create_model_invocation_job(
            modelId=metadata['model'],
            jobName=f"batch-{batch_path.name}",
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": s3_input_uri
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": f"s3://{output_bucket}/batch_outputs/{batch_path.name}/"
                }
            },
            roleArn=role_arn
        )
        
        # Update metadata with job information
        metadata['status'] = 'submitted'
        metadata['job_arn'] = response['jobArn']
        metadata['s3_input_uri'] = s3_input_uri
        metadata['s3_output_uri'] = f"s3://{output_bucket}/batch_outputs/{batch_path.name}/"
        
        with open(batch_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Move batch to submitted directory
        submitted_path = Path('batch_data/submitted') / batch_path.name
        batch_path.rename(submitted_path)
        
        print(f"Successfully submitted batch job: {response['jobArn']}")
        
    except ClientError as e:
        print(f"Error creating batch job: {e}")
        # Update metadata with error
        metadata['status'] = 'error'
        metadata['error'] = str(e)
        with open(batch_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Process pending Bedrock batch jobs')
    parser.add_argument('--input-bucket', required=True, help='S3 bucket for input data')
    parser.add_argument('--output-bucket', required=True, help='S3 bucket for output data')
    parser.add_argument('--region', help='AWS region (default: from environment)')
    parser.add_argument('--batch-dir', default='batch_data', help='Base directory for batch data')
    args = parser.parse_args()
    
    # Initialize clients
    bedrock_client = init_bedrock_client(args.region)
    s3_client = boto3.client('s3')
    
    # Process all pending batches
    pending_dir = Path(args.batch_dir) / 'pending'
    if not pending_dir.exists():
        print(f"No pending directory found at {pending_dir}")
        return
        
    for batch_path in pending_dir.iterdir():
        if batch_path.is_dir():
            try:
                process_pending_batch(
                    bedrock_client,
                    s3_client,
                    batch_path,
                    args.input_bucket,
                    args.output_bucket
                )
            except Exception as e:
                print(f"Error processing batch {batch_path}: {e}")
                continue

if __name__ == "__main__":
    main() 