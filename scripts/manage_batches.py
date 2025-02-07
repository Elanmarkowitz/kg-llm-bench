#!/usr/bin/env python3
import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Optional
from llm.batch_manager import BatchManager

def force_new_batches():
    """Force all active providers to start new batches."""
    manager = BatchManager()
    providers = manager.get_all_providers()
    
    if not providers:
        print("No active batch providers found")
        return
    
    for model, provider in providers.items():
        if provider.current_batch_size > 0:
            print(f"Forcing new batch for model {model}")
            provider._start_new_batch()
        else:
            print(f"No active batch for model {model}")

def show_batch_status():
    """Show status of all active batches."""
    manager = BatchManager()
    providers = manager.get_all_providers()
    
    if not providers:
        print("No active batch providers found")
        return
    
    print("\nActive Batches:")
    print("-" * 80)
    for model, provider in providers.items():
        print(f"\nModel: {model}")
        if provider.current_batch:
            print(f"Current Batch: {provider.current_batch}")
            print(f"Records: {provider.current_batch_size}")
            print(f"Path: {provider.current_batch_path}")
        else:
            print("No active batch")
    
    # Show pending batches
    pending_dir = Path('batch_data/pending')
    if pending_dir.exists():
        print("\nPending Batches:")
        print("-" * 80)
        for batch_dir in pending_dir.iterdir():
            if batch_dir.is_dir():
                metadata_file = batch_dir / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    print(f"Batch: {batch_dir.name}")
                    print(f"  Model: {metadata.get('model', 'unknown')}")
                    print(f"  Records: {metadata.get('record_count', 0)}")
                else:
                    print(f"Batch: {batch_dir.name} (no metadata)")
    
    # Show submitted batches
    submitted_dir = Path('batch_data/submitted')
    if submitted_dir.exists():
        print("\nSubmitted Batches:")
        print("-" * 80)
        for batch_dir in submitted_dir.iterdir():
            if batch_dir.is_dir():
                print(f"Batch: {batch_dir.name}")

def find_result_files(batch_metadata: dict) -> List[Path]:
    """Find all result files that contain records from this batch."""
    result_files = set()
    benchmark_dir = Path('benchmark_data')
    
    # Scan all result files
    for result_file in benchmark_dir.rglob('*_results.json'):
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # Check if any result contains a pending response for this batch
            for instance in results:
                if instance and isinstance(instance.get('response'), str):
                    response = instance['response']
                    if response.startswith('PENDING:'):
                        record_id = response.split(':')[1]
                        if record_id in batch_metadata.get('record_map', {}):
                            result_files.add(result_file)
                            break
        except (json.JSONDecodeError, IOError):
            continue
    
    return list(result_files)

def delete_pending_batch(batch_id: Optional[str] = None, delete_all: bool = False):
    """Delete pending batch(es) and clean up result files."""
    pending_dir = Path('batch_data/pending')
    if not pending_dir.exists():
        print("No pending batches directory found")
        return
    
    batches_to_delete = []
    if delete_all:
        batches_to_delete = list(pending_dir.iterdir())
    elif batch_id:
        batch_path = pending_dir / batch_id
        if batch_path.exists():
            batches_to_delete = [batch_path]
        else:
            print(f"Batch {batch_id} not found")
            return
    else:
        print("Must specify either --batch-id or --all")
        return
    
    for batch_path in batches_to_delete:
        if not batch_path.is_dir():
            continue
            
        print(f"\nProcessing batch: {batch_path.name}")
        
        # Load metadata
        metadata_file = batch_path / 'metadata.json'
        if not metadata_file.exists():
            print("No metadata file found, skipping...")
            continue
            
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Find result files containing records from this batch
        result_files = find_result_files(metadata)
        
        if result_files:
            print("\nFound result files to update:")
            for result_file in result_files:
                print(f"  {result_file}")
            
            update_results = input("\nUpdate result files? [y/N] ").lower() == 'y'

            if update_results:
                for result_file in result_files:
                    try:
                        # Load results
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        # Remove pending responses for this batch
                        for instance in results[:]:  # Iterate over a copy of the list
                            if instance and isinstance(instance.get('response'), str):
                                response = instance['response']
                                if response.startswith('PENDING:'):
                                    record_id = response.split(':')[1]
                                    if record_id in metadata.get('record_map', {}):
                                        results.remove(instance)  # Delete the instance
                        
                        if results:  # Check if there are any results left
                            # Save updated results
                            with open(result_file, 'w') as f:
                                json.dump(results, f, indent=2)
                            print(f"Updated {result_file}")
                        else:
                            # If no results left, delete the file
                            os.remove(result_file)
                            print(f"Deleted empty result file: {result_file}")
                    except Exception as e:
                        print(f"Error updating {result_file}: {e}")
        
        if not update_results:
            continue_delete = input("Results were not deleted. Do you want to continue with batch deletion? [y/N] ").lower() == 'y'
            if not continue_delete:
                print("Batch deletion aborted.")
                return
        try:
            shutil.rmtree(batch_path)
            print(f"Deleted batch directory: {batch_path}")
        except Exception as e:
            print(f"Error deleting {batch_path}: {e}")
    
    print("\nBatch cleanup complete")

def main():
    parser = argparse.ArgumentParser(description='Manage Bedrock batch processing')
    parser.add_argument('--force-new', action='store_true', 
                       help='Force all providers to start new batches')
    parser.add_argument('--status', action='store_true',
                       help='Show status of all active batches')
    parser.add_argument('--delete-batch', metavar='BATCH_ID',
                       help='Delete a specific pending batch')
    parser.add_argument('--delete-all', action='store_true',
                       help='Delete all pending batches')
    args = parser.parse_args()
    
    if args.force_new:
        force_new_batches()
    
    if args.status:
        show_batch_status()
    
    if args.delete_batch or args.delete_all:
        delete_pending_batch(args.delete_batch, args.delete_all)
    
    if not (args.force_new or args.status or args.delete_batch or args.delete_all):
        parser.print_help()

if __name__ == "__main__":
    main() 