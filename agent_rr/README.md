# AgentRR

# Prepare Environment

```bash
pip install -r requirements-agentrr.txt
```

# Train Latent Memory Model

## Data Preparation

Step 1: Prepare JSON task templates and store in `templates.json`. For example, see `train/train_data_example/wechat/templates.json`. 

Step 2: Create train/test dataset based on task templates using the following command:

```bash
python -m train.prepare_data --task both --train_path <output path of the train split> --test_path <output path of the test split>
```

Step 3: Train Embedding and Reranker model with [ms-swift](https://github.com/modelscope/ms-swift), see official training example [SWIFT](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md).

# Run Experiment

```bash
python run_experiment.py --data_path <path to the test split> --embedder_path <path to the embedding model> --reranker_path <path to the reranker model>
```