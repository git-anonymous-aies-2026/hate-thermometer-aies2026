# Measuring the Measurer: Psychometric Evaluation of LLMs as Hate Speech Annotators

We apply the Many-Facet Rasch Model (MFRM) to evaluate five frontier LLMs as hate speech annotators, benchmarked against 591 human raters from the Measuring Hate Speech corpus.

📄 Papers : Submitted | AIES 2026 | Under Review

## Key Findings 

- Systematic leniency: All LLMs except GPT-5 are more lenient than the average human rater, clustering within 2.0 logits — less than 25% of the human severity range
- Severity-dependent bias: LLMs underestimate mild/implicit hate while overestimating serious hate — concentrating error where early-stage intervention matters most
- Prompting strategy: Modular prompting induces greater leniency than monolithic but preserves dimensional ordering — monolithic is the more cost-effective choice
- Cultural convergence: DeepSeek (non-Western) converges with Western models under monolithic prompting but shows disproportionate sensitivity under modular prompting
- Calibration paradox: Despite leniency as raters, most LLMs produce higher calibrated hate scores than humans after joint MFRM correction

## Dataset Used 
We use the Measuring Hate Speech corpus by Kennedy et al. (2020). See [MHS Corpus on HugginFace](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech).
```
from datasets import load_dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
```

## Models used
    | Model               | Provider   | Type   | Origin      |
    |---------------------|------------|--------|-------------|
    | GPT-5               | OpenAI     | Closed | Western     |
    | GPT-5-mini          | OpenAI     | Closed | Western     |
    | Claude Haiku 4.5    | Anthropic  | Closed | Western     |
    | Gemini 2.5 Flash    | Google     | Closed | Western     |
    | DeepSeek-V4         | DeepSeek   | Open   | Non-Western |

## Setup

### Clone the repository

```
git clone https://github.com/yourusername/aies_2026.git

cd aies_2026
```

### RaschPy

This repository includes a local fork of [RaschPy](https://github.com/MarkElliott999/RaschPy).
See `RaschPy/` for the modified source.

### Read and activate environment

```
conda create -n hate-speech-llm python=3.11
conda activate hate-speech-llm
pip install -r requirements.txt
```

### configure API Keys
```
cp .env.example .env
```

edit the `.env` with your personal API keys

```
ANTHROPIC_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
DEEPSEEK_API_KEY=your-key-here
TOGETHER_API_KEY=your-key-here
CMU_OPENAI_API_KEY=your-key-here
```
### Path setup

Edit `config.py` and set the `BASE_DIR` to your local file directory path
```
BASE_DIR = Path("/your/local/path/to/final_AIES_2026")
```
### Slur Database

The slur database (`rsdb_slurs.json`) is not included in this 
repository. To generate it run:

```bash
python Annotation/scrape_rsdb.py
```

This scrapes the Racial Slur Database and saves it locally.

### Run annotation

Update the model confidureagtion in `lm`
Run all annotation(Mono then Modu) for that model using
```
python aies_2026/run_experiment.py
```

### Run MFRM anlaysis (Plots, figures and stats)

Import all function from the   `MFRM_analysis.py`, Implement them in the `MFRM_AIES26.ipynb`


## Repo Set up

    Main_folder
    │
    AIES_2026/                     # Paper 1 — AIES 2026
    ├── README.md
    ├── config.py                   # Paths, dimensions, colors, model maps
    ├── requirment.py               # Single entry point
    ├── .env.examples               # API key used and setup
    ├── .gitignore
    │
    ├── annotation/
    │   ├── experiment.py           # Main annotation runner
    │   ├── signatures.py           # DSPy signature definitions
    │   ├── dimension_rating.py     # Monolithic and modular raters
    │   ├── majority_vote.py        # Majority voting functions
    │   ├── scrape_rsbd.py          # constructing the slur tagging database
    │   ├── patching.py             # Incomplete run patching
    │   └── utils.py                # Helper functions
    │
    ├── analysis/
    │   └── mfrm_analysis.py        # MFRM calibration pipeline
    │  
    ├── RaschPy/
    │   ├── RaschPy/__init__.py
    │   └── setup.py                # local fork
    │
    └── notebooks/
        ├── MFRM_AIES26.ipynb        # analysis notebook
        ├── MFRM_AIES26              # MFRM calibration folder
        ├── MFRM_output_files        # All data necessary
        └── example_gpt4o            # .csv example show slu tagging, mono and modu rating and reasoning

## Citation
If you use this work please cite:

## Acknowledgements

- Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application. [Kennedy et al. 2022](https://arxiv.org/abs/2009.10277)
- DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. [Khattab et al. 2023](https://arxiv.org/abs/2310.03714)
- [RaschPy](https://github.com/MarkElliott999/RaschPy) for the open-source Python MFRM implementation