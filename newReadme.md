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

## Instructions to download data from DVC


## Instructions to generate new data from KG

To generate new data from the knowledge graph, follow these steps:

### Step 1: Construct Base Datasets
Use the `construct_base_datasets.py` script to create base datasets for knowledge graph tasks. This script loads the knowledge graph and generates datasets based on the task configurations.

Run the script with the following command:
```bash
python construct_base_datasets.py --config configs/construct_base_datasets_small.yaml
```
Ensure that the configuration file (configs/construct_base_datasets_small.yaml) is properly set up with the desired task configurations.


### Step 2: Construct Formatted Datasets
After constructing the base datasets, use the `construct_formatted_datasets.py` script to format the datasets for specific tasks. This script processes the base datasets and applies formatting based on the conversion and pseudonymizer configurations.

Run the script with the following command:
```bash
python construct_formatted_datasets.py --config configs/construct_formatted_datasets_small.yaml
```

Ensure that the configuration file (`configs/construct_formatted_datasets_small.yaml`) is properly set up with the desired conversion and pseudonymizer configurations.



## Instructions to run experiments with models

To run experiments with the models, use the `run_experiments.py` script. This script executes tasks defined in the configuration file and evaluates the performance of various models on knowledge graph tasks.

```bash
python run_experiments.py --config configs/run_small_datasets.yaml
```

### Optional Arguments
- `--reevaluate`: Reevaluate existing results.
- `--reevaluate_only`: Only reevaluate existing results without running new experiments.
- `--batch`: Enable batch mode for AWS Bedrock models.

For example, to reevaluate results in batch mode:
```bash
python run_experiments.py --config configs/run_small_datasets.yaml --reevaluate --batch
```

Ensure that the configuration file (`configs/run_small_datasets.yaml`) is properly set up with the desired task, pseudonymizer, and conversion configurations.
```