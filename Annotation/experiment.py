import os, json, time, gc, subprocess, krippendorff, psutil, re
import pandas as pd
import numpy as np
import RaschPy as rp
#dspy
import warnings, sys
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
import random
random.seed(42)

from tqdm import tqdm
from datetime import datetime
from krippendorff import alpha as krippendorff_alpha
from datasets import Dataset, load_dataset

from dimension_rating import (
    RatingNineDimensionsTogether,
    RatingNineDimensionsSeparate
)

from hate_classification import HateClassificationComparison

from signatures import HateDefSlurOnly

from patching import (
    patch_incomplete_runs, 
    validate_completeness, 
    # fix_incomplete_runs
)
from majority_vote import (
    get_llm_ratings_with_majority_vote, 
    extracting_data_for_llm_inference,
    get_llm_ratings_batching,
    get_majority_vote_from_dict
)
from utils import (
    apply_slur_tags_placeholder,
    load_slur_database,
    convert_to_dspy_example,
    # get_llm_label
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import config
from config import DIMENSIONS, DATA_DIR, DATA_DIR_RAW, DATA_DIR_MFRM

from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error, 
                             precision_recall_fscore_support, 
                             confusion_matrix, 
                             classification_report, 
                             accuracy_score)
from scipy.stats import pearsonr, spearmanr

from dotenv import load_dotenv
import dspy
from dspy.predict import Predict
#addedd this to avoid memory leaks and overheating issues
gc.collect()
if hasattr(dspy, 'settings'):
    dspy.settings.configure(trace=[])
import importlib

import litellm
litellm.cache = None

load_dotenv()
CMU_OPENAI_API_KEY = os.environ.get("CMU_OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
ALLIBABA_API_KEY = os.environ.get("ALLIBABA_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")


print("Loaded key:", DEEPSEEK_API_KEY[:10] if DEEPSEEK_API_KEY else None)
print()
lm = dspy.LM(model="openai/deepseek-v4-flash",
             api_key=DEEPSEEK_API_KEY,
             api_base= "https://api.deepseek.com",  
             cache=False)

dspy.configure(lm=lm)
print("="*80)
print("STEP 0: TESTING MODEL AND CONFIGURATION")
print("="*80)
print(f"\nConfigured model: {dspy.settings.lm.model}")
print('Test that llm model is working. Ask:What is the capital of France?')
response = lm("what is the capital of France?")
print(response)
print()

num_samples = 2
num_runs = 1
llm_full_name = dspy.settings.lm.model
llm_model_name = str(llm_full_name).split('/')[-1] # Extract model name only

def main(seed=42):
    # Get LLM model name
    llm_model_name = str(llm_full_name).split('/')[-1]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ==========================================================================
    # STEP 1: Load and Sample Dataset
    # ==========================================================================
    print("="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    berkeley_dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    dataset = berkeley_dataset['train']
    random.seed(seed) 
    unique_texts = list(dict.fromkeys(dataset['text']))
    sample_texts = random.sample(unique_texts, num_samples)

    # ==========================================================================
    # STEP 2: Filter and Sample
    # ==========================================================================
    print("\n" + "="*80)
    print("STEP 2: FILTERING AND SAMPLING")
    print("="*80)
    # Human dataset (keep duplicates for MFRM) 
    human_sample_data = dataset.filter(lambda x: x['text'] in set(sample_texts))
    human_irt_dataset = human_sample_data.to_pandas()

    # Drop duplicates based strictly on the comment_id so only ONE version of the text survives
    llm_inference_dataset = human_irt_dataset.drop_duplicates(subset=['comment_id'])
    llm_input_sample = llm_inference_dataset[['text', 'comment_id']]
    #convert back to HF for rest of implementation
    sample_data = Dataset.from_pandas(
        llm_input_sample,
        preserve_index=False  # Don't include pandas index
    )

    print(f"Loaded {len(sample_texts)} unique texts.")
    print(f"\nHuman IRT Dataset rows (with overlapping reviewers): {len(human_irt_dataset)}")
    print(f"\nLLM Input Dataset rows (strictly deduplicated): {len(sample_data)}")
    print()

    # # ==========================================================================
    # # STEP 3: Slur Tagging
    # # ==========================================================================
    print("\n" + "="*80)
    print("STEP 3: SLUR TAGGING")
    print("="*80)
    slur_database_sorted = load_slur_database()

    slur_tagged_texts = [
        apply_slur_tags_placeholder(example['text'], slur_database_sorted)
        for example in sample_data
    ]
    sample_data = sample_data.add_column("slur_tagged_text", slur_tagged_texts)
    print("\nSlur tagging Completed...\n")

    # # ==========================================================================
    # # STEP 4: Initialize Raters
    # # ==========================================================================
    print("\n" + "="*80)
    print("STEP 4: INITIALIZING RATERS")
    print("="*80)
    rater_mono = RatingNineDimensionsTogether()
    rater_mod = RatingNineDimensionsSeparate()
    
    # ==========================================================================
    # STEP 5: Extract Gold Labels
    # ==========================================================================
    print("\n" + "="*80)
    print("STEP 5: EXTRACTING GOLD LABELS")
    print("="*80)
    gold_dataset = extracting_data_for_llm_inference(
        llm_inference_dataset, 
        human_irt_dataset)
    save_gold = f'gold_kennedy_{timestamp}.csv'
    gold_csv_file = os.path.join(DATA_DIR, save_gold)
    gold_dataset.to_csv(gold_csv_file, index=False)
    print(f"\nSaved gold labels as CSV to: {gold_csv_file}")

    # ==========================================================================
    # STEP 6 — LLM annotation (run both strategies)
    # ==========================================================================

    print("="*80 + "\nSTEP 6 & 7: RUNNING LLM ANNOTATION\n" + "="*80)


    for strategy, rater in  [('mono', rater_mono), ('modu', rater_mod)]:
        run_label = f'{llm_model_name}_{strategy}'
        print(f"\n--- Running {run_label} ---")

        # Annotate
        results  = get_llm_ratings_with_majority_vote(
            sample_data,
            rater_dimension = rater,
            n_runs          = num_runs,
            llm_model_name  = run_label
        )
        raw_path = os.path.join(DATA_DIR_RAW, f'{run_label}_unpatched_{timestamp}.csv')
        print('results head for the majority vote:\n', results)
        results.to_csv(raw_path, index=False)
        print(f"\nSaved unpatched: {raw_path}")

        print(f"\nValidating {run_label}...")
        is_complete = validate_completeness(results, n_runs=num_runs)
        if not is_complete:
            print("Patching incomplete runs...")
            results = patch_incomplete_runs(
                results,
                llm_model_name  = run_label,
                rater_dimension = rater,
                n_runs          = num_runs
            )
            validate_completeness(results, n_runs=num_runs)
        else:
            print("\nAll runs complete, no patching needed.")

        # Save final results regardless of whether patching was needed
        patched_path = os.path.join(
                DATA_DIR, f'{run_label}_patched_{timestamp}.csv'
            )
        results.to_csv(patched_path, index=False)
        print(f"\nSaved final results: {patched_path}\n")

        # Save MFRM-ready format
        mfrm_ready      = results[['annotator_id', 'comment_id'] + DIMENSIONS].copy()
        mfrm_ready_path = os.path.join(
            DATA_DIR_MFRM, f'mfrm_{run_label}_patched_{timestamp}.csv'
        )
        mfrm_ready.to_csv(mfrm_ready_path, index=False)
        print(f"\nSaved MFRM ready: {mfrm_ready_path}")

    print("\n\nAll experiments complete!")
if __name__ == "__main__":
    main()
