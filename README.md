# MisCalibrated Judges: Psychometric Evaluation of LLMs as Hate Speech Annotators

We apply the Many-Facet Rasch Model (MFRM) to evaluate five frontier LLMs as hate speech annotators, benchmarked against 591 human raters from the Measuring Hate Speech corpus.

📄 Papers : Submitted | AISI 2026 | Under Review

## Key Findings
### General 
- Most LLMs are systematically more lenient than the average human annotator, yet cluster far more tightly together as a group than human crowdworkers do.
- LLMs tend to underestimate mild/implicit hate while overestimating overt hate — concentrating annotation error exactly where early-stage moderation intervention matters most.
- Modular prompting induces greater leniency than monolithic prompting, but the relative ranking of dimension difficulty and comment severity is preserved either way — making monolithic the more cost-effective choice at comparable quality.
- Human inter-rater reliability is not a reliable proxy for LLM cross-model consistency.
- DeepSeek (non-Western) converges with the Western models evaluated here under monolithic prompting but shows disproportionate sensitivity under modular prompting.
- RaschPy (Python) reproduces Winsteps-comparable MFRM calibration (r = 0.819 vs. Kennedy et al.'s gold-standard scores), offering a free, programmatic alternative for researchers without access to proprietary psychometric software.
- [RaschPy](https://github.com/MarkElliott999/RaschPy) reproduces Winsteps-comparable MFRM calibration (r = 0.819 vs. [Kennedy et al. 2022](https://arxiv.org/abs/2009.10277)'s gold-standard scores), offering a free, programmatic alternative for researchers without access to proprietary psychometric software.

### Robustness Checks
- Removing the external slur-definition preprocessing step leaves human-LLM dimensional correlations essentially unchanged, confirming the core findings aren't an artifact of the slur database.
- Post-hoc analysis of incomplete annotations found no evidence of systematic safety-driven refusal; incompleteness mostly attributable to API infrastructure failures and pipeline branching (non-targeted comments).

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

cd hate-thermometer-anonymous
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
BASE_DIR = Path("/your/local/path/to/hate-thermometer-anonymous")
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

Import all function from the   `MFRM_analysis.py`, Implement them in the `MFRM_AISI26.ipynb`


## Repo Set up

    Main_folder
    │
    hate-thermometer-anonymous/     # AISI 2026
    ├── README.md
    ├── config.py                   # Paths, dimensions, colors, model maps
    ├── requirment.txt              # Single entry point
    ├── .env                        # API key used and setup
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
    |
    ├── hate_thermometer_data/
    │   └── data                    #Original outputs. data_outputs_ablate = outomes from no slur taggin expeirment
    |        ├── mfrm_data          # Final data converted into MFRM acceptable format for Psychometric analysis 
    │        ├── patched_data       # Final LLM ratting outcome after patching(rerun to reannotate incomplete annotation) 
    │        └── raw_data           # Raw original LLM Outcome (Run 1)
    |
    ├── analysis/
    │   └── mfrm_analysis.py        # MFRM calibration pipeline
    │  
    ├── RaschPy_aisi/
    │   ├── RaschPy/__init__.py
    │   └── setup.py                # local fork
    │
    └── notebooks/
        ├── MFRM_AISI26.ipynb        # analysis notebook
        ├── refusal_analysis.ipynb   # analysis of LLM non annotations
        ├── ablation_noslur          # no slur tagging experiment with deepseek and gemini
        └── example_gpt4o            # .csv example show slu tagging, mono and modu rating and reasoning

## Citation
If you use this work please cite:

## Acknowledgements

- Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application. [Kennedy et al. 2022](https://arxiv.org/abs/2009.10277)
- DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. [Khattab et al. 2023](https://arxiv.org/abs/2310.03714)
- [RaschPy](https://github.com/MarkElliott999/RaschPy) for the open-source Python MFRM implementation