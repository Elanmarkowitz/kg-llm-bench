# Knowledge Graph LLM Benchmark with Batch Processing

This repository contains a benchmarking system for testing LLM performance on knowledge graph tasks, with support for AWS Bedrock batch processing.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_region
```

3. Set up S3 buckets for batch processing:
```bash
export BEDROCK_INPUT_BUCKET=your-input-bucket
export BEDROCK_OUTPUT_BUCKET=your-output-bucket
```

4. Create an IAM role for Bedrock batch processing with appropriate permissions and set:
```bash
export BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::account:role/role-name
```

## Running Experiments with Batch Processing

1. Run experiments in batch mode:
```bash
python run_experiments.py --config configs/run_small_datasets.yaml --batch
```

This will:
- Create batch records in `batch_data/pending/`
- Store placeholder results with `PENDING:` status
- Continue processing all tasks

2. Submit batch jobs to Bedrock:
```bash
python scripts/process_batches.py \
  --input-bucket $BEDROCK_INPUT_BUCKET \
  --output-bucket $BEDROCK_OUTPUT_BUCKET
```

3. Collect and process results:
```bash
python scripts/collect_results.py
```

Run this periodically to:
- Check job status
- Download completed results
- Update task result files
- Move completed batches to archive

## Directory Structure

```
.
├── batch_data/
│   ├── pending/      # Batches waiting to be submitted
│   ├── submitted/    # Batches currently processing
│   └── completed/    # Processed batches with results
├── benchmark_data/   # Task data and results
├── configs/         # Configuration files
├── llm/            # LLM provider implementations
├── scripts/        # Utility scripts
└── tasks/          # Task implementations
```

## Batch Processing Flow

1. **Accumulation Phase**
   - BatchBedrock provider accumulates requests
   - Creates JSONL files with proper format
   - Stores metadata for tracking

2. **Submission Phase**
   - Uploads records to S3
   - Creates Bedrock batch jobs
   - Tracks job ARNs and status

3. **Collection Phase**
   - Monitors job completion
   - Downloads and processes results
   - Updates original task results

## Configuration

The batch processing behavior can be configured through environment variables or command line arguments:

- `BATCH_SIZE`: Maximum records per batch (default: 100)
- `BATCH_TIMEOUT`: Minutes before starting new batch (default: 60)
- `AWS_DEFAULT_REGION`: AWS region for Bedrock
- `BEDROCK_INPUT_BUCKET`: S3 bucket for input data
- `BEDROCK_OUTPUT_BUCKET`: S3 bucket for output data
- `BEDROCK_BATCH_ROLE_ARN`: IAM role ARN for batch jobs

## Error Handling

The system includes robust error handling:
- Failed jobs are marked and preserved
- Partial results are processed when available
- Automatic retries for transient failures
- Detailed logging for debugging

## Monitoring

Monitor batch processing status:
- Check `batch_data/*/metadata.json` files
- View AWS Bedrock console
- Monitor S3 buckets
- Check task result files

## Best Practices

1. Use appropriate batch sizes (100-1000 records)
2. Monitor costs and completion times
3. Regularly collect results
4. Maintain S3 bucket lifecycle policies
5. Review and archive completed batches

## Troubleshooting

Common issues and solutions:
1. **Missing Results**: Check job status and S3 paths
2. **Failed Jobs**: Review CloudWatch logs
3. **Stuck Jobs**: Check IAM permissions
4. **S3 Errors**: Verify bucket permissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details 