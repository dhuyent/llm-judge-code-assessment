# llm-judge-code-assessment

Evaluating LLM-as-a-Judge for Programming Problem Assessment Using Problem Descriptions and Submissions.

## Overview

This project is implemented as a pipeline of Jupyter Notebooks in the `src/` directory.  
The pipeline generates buggy submissions, asks LLMs to judge them, classifies the results across multiple runs, and filters difficult cases for later stages.

## Requirements

- Python 3.12
- Jupyter Notebook / JupyterLab
- `pandas`
- `numpy`
- `openai` and/or `anthropic` only for notebooks that call model APIs

Install dependencies locally:

```bash
pip install -r requirements.txt
```

or:

```bash
pip install pandas numpy jupyter
pip install openai anthropic   # only needed for API-based judge notebooks
```

Kaggle Notebook is recommended because Python, Jupyter, pandas, and numpy are already available.

## API Keys

API keys are only required for `judge_*.ipynb`.

On Kaggle, store keys in **Add-ons > Secrets**.  
On a local machine, set the corresponding environment variables, for example:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

The `classify`, `filter`, and `filter2` notebooks do not require API keys.

## Running the Pipeline

Before running each notebook:

1. Open the `CONFIG` cell at the beginning of the notebook.
2. Update input/output paths for your environment.
3. Run all cells from top to bottom.

Recommended execution order:

### 1. Generate buggy submissions

Run:

```text
src/create-buggy.ipynb
```

This creates buggy submissions together with ground-truth fields such as `gt_status` and `gt_input`.

### 2. Run LLM judges

Run the judge notebook for each model, for example:

```text
src/judge_claude-opus-4-8.ipynb
src/judge_qwen3.7-max.ipynb
```

`judge.ipynb` is the shared/original judge notebook.

Each model should be run multiple times to measure stability:

- Stage 1: 3 runs per model
- Stage 2: 2 runs per model

Example output:

```text
step2_logic1_<model>_1.csv
step2_logic1_<model>_2.csv
step2_logic1_<model>_3.csv
```

The experiment groups are:

```text
logic1
logic2
reference1
```

### 3. Classify results

Run:

```text
src/classify.ipynb
```

For each submission ID, the notebook compares `pred_status` with `gt_status` across repeated runs and assigns:

- **Label 1**: all runs are correct
- **Label 2**: some, but not all, runs are correct
- **Label 3**: no run is correct

Example output:

```text
step1_classify1_<group>.csv
```

### 4. Filter difficult cases

For the standard filtering step, run:

```text
src/filter.ipynb
```

This keeps difficult cases with Label 2 or Label 3 after manual recheck.

Example output:

```text
step1_filter1_<group>.csv
```

For the cross-model filtering step, run:

```text
src/filter2.ipynb
```

This keeps the intersection of IDs that are difficult for both models, i.e. Label 2 or Label 3 for both models.

Configure:

```text
INPUT_DIR
GROUPS
RUNS
FNAME
```

Example output:

```text
step1_filter2_<group>.csv
```

## Pipeline Summary

```text
create-buggy.ipynb
        |
        v
buggy submissions + ground truth
        |
        v
judge_<model>.ipynb
        |
        v
multiple prediction CSV files
        |
        v
classify.ipynb
        |
        +-------------------+
        |                   |
        v                   v
 filter.ipynb          filter2.ipynb
(Label 2 & 3)     (intersection across models)
```

## Notes

- Default notebook paths may point to Kaggle directories such as `/kaggle/input/...`.
- When running locally, update paths in each notebook's `CONFIG` cell.
- On Kaggle, generated files are typically written to `/kaggle/working/`.
- GPU is not required for the CSV-processing stages.