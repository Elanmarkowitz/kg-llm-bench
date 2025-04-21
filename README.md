# Knowledge Graph LLM Benchmark with Batch Processing

This repository contains a benchmarking system for testing LLM performance on knowledge graph tasks, with support for AWS Bedrock batch processing.

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for the list of dependencies.


## Setup

Follow these steps to set up the project:

1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Elanmarkowitz/kg-llm-bench.git
   cd kg-llm-bench
   ```

2. **Create a Virtual Environment (Optional)**  
   Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**  
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**  
   This project uses `python-dotenv` to manage environment variables. Create a `.env` file in the root directory and add your AWS credentials, OpenAI API and Gemini API keys:
   ```env
   AWS_ACCESS_KEY_ID=<your-access-key-id>
   AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
   AWS_REGION=<your-region>
   OPENAI_API_KEY=<your-openai-api-key>
   GEMINI_API_KEY=<your-gemini-api-key>
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