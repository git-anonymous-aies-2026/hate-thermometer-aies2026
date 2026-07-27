"""
LLM majority voting annotation
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import dspy
import json

from utils import convert_labels_to_numeric, get_majority_vote_from_dict
from config import DIMENSIONS


def extract_gold_labels(dataset_sample):
    """
    Extract human annotations from Kennedy dataset.
    """
    gold_labels = []
    for example in dataset_sample:
        gold_labels.append({
            'text': example['text'], #text since we want the gold
            'annotator_id':example['annotator_id'],
            'comment_id':example['comment_id'],
            'sentiment': example['sentiment'],
            'respect': example['respect'],
            'insult': example['insult'],
            'humiliate': example['humiliate'],
            'status': example['status'],
            'dehumanize': example['dehumanize'],
            'violence': example['violence'],
            'genocide': example['genocide'],
            'attack_defend':example['attack_defend'],
            'hatespeech':example['hatespeech']
        })
    gold_labels = pd.DataFrame(gold_labels)
    print('The Kennedy Gold Labels:')
    print(gold_labels.head())
    return gold_labels

def extracting_data_for_llm_inference(llm_inference_dataset, human_irt_dataset, seed=42):
    """
    Extract gold standard labels from the human annotations.
    For each unique comment in llm_inference_dataset, get the corresponding
    human annotations from human_irt_dataset.
    """
    np.random.seed(seed)
    
    # Also include any pre-computed scores if available
    additional_columns = [
        'hate_speech_score', 'annotator_severity', 'annotator_infitms', 
        'annotator_outfitms'
    ]
    
    # Get all comment_ids from LLM inference dataset
    target_comment_ids = set(llm_inference_dataset['comment_id'].unique())
    
    print(f"\nExtracting gold labels for {len(target_comment_ids)} comments...")
    # Filter human data to only include our target comments
    gold_data = human_irt_dataset[
        human_irt_dataset['comment_id'].isin(target_comment_ids)
    ].copy()

    print(f"Found {len(gold_data)} human annotations")
    
    # Keep relevant columns
    base_columns = ['text', 'comment_id', 'annotator_id'] + DIMENSIONS

    available_additional = [
        col for col in additional_columns 
        if col in gold_data.columns
    ]
    
    columns_to_keep = base_columns + available_additional
    
    # Filter to existing columns only
    existing_columns = [
        col for col in columns_to_keep 
        if col in gold_data.columns
    ]
    
    gold_labels = gold_data[existing_columns].copy()
    
    # Sort by comment_id and annotator_id for easier viewing
    gold_labels = gold_labels.sort_values(['comment_id', 'annotator_id']).reset_index(drop=True)
    
    print(f"  Extracted columns: {gold_labels.columns.tolist()}")
    print(f"  \nTotal rows: {len(gold_labels)}")
    print(f"  Unique comments: {gold_labels['comment_id'].nunique()}")
    print(f"  Unique annotators: {gold_labels['annotator_id'].nunique()}")
    
    return gold_labels


def get_llm_ratings_with_majority_vote(dataset_samples, rater_dimension, n_runs, llm_model_name):
    
    # Invalid values
    INVALID_VALUES = {'', 'none', 'null', 'n/a'}
    
    dataset_df = pd.DataFrame(dataset_samples)
    num_examples = len(dataset_df)

    # Initialize results storage
    results_by_index = {
        idx: {dim: [] for dim in DIMENSIONS}
        for idx in range(num_examples)
    }
    
    print(f"\nRunning LLM annotation: {n_runs} runs × {num_examples} samples = {n_runs * num_examples} total calls")
    total_iterations = n_runs * num_examples

    with tqdm(total=total_iterations, desc="Run Numbers", unit="sample") as pbar:
        for run_idx in range(n_runs):
            # Shuffle with different seed each run
            shuffled_df = dataset_df.sample(
                frac=1, 
                random_state=run_idx
            ).reset_index(drop=False)

            for row in shuffled_df.itertuples():
                text = row.text
                slur_tagged_text = row.slur_tagged_text
                original_idx = row.index
                try:
                    dim_result = rater_dimension(text=text, slur_tagged_text=slur_tagged_text)
                    # hate_result = rater_hate(text=text) #actually already in the dim_results, no need to call it again
                    dim_ratings = {}
                    for dim in DIMENSIONS[:-1]:
                        raw_value = getattr(dim_result, dim, None)
                        # Quick validation
                        if raw_value is None:
                            rating = None
                        elif isinstance(raw_value, (int, float)):
                            rating = int(raw_value)
                            rating = rating if 0 <= rating <= 4 else None
                        else:
                            # String validation
                            val_str = str(raw_value).strip().lower()
                            if val_str in INVALID_VALUES:
                                rating = None
                            else:
                                try:
                                    rating = int(raw_value)
                                    rating = rating if 0 <= rating <= 4 else None
                                except (ValueError, TypeError):
                                    rating = None
                        results_by_index[original_idx][dim].append(rating)
                    
                    # Extract hatespeech
                    raw_hate    = getattr(dim_result, 'hatespeech', None)
                    numeric     = convert_labels_to_numeric(raw_hate)
                    hate_rating = int(numeric) if numeric is not None else None
                    results_by_index[original_idx]['hatespeech'].append(hate_rating)


                except Exception as e:
                    # Log error but continue
                    print(f"\n⚠️ Error on idx {original_idx+1}: {type(e).__name__}")
                    print(f"   Text: {text[:50]}...")
                    
                    # Append None for all dimensions on failure
                    for dim in DIMENSIONS:
                        results_by_index[original_idx][dim].append(None)

                
                # Update progress bar
                pbar.update(1)
            
    print("\nCalculating majority votes...")
    
    final_outputs = []
    
    for original_idx in range(num_examples):
        
        original_example = dataset_df.iloc[original_idx]
        ratings_across_runs = results_by_index[original_idx]
        
        majority_votes = {}
        vote_distributions = {}
        
        for dim in DIMENSIONS:
            # Get valid ratings (filter None)
            valid_ratings = [
                r for r in ratings_across_runs[dim] 
                if r is not None
            ]
            
            if len(valid_ratings) == 0:
                majority_votes[dim] = None
                vote_distributions[dim] = {}
            else:
                vote_counts = dict(Counter(valid_ratings))
                majority_votes[dim]     = get_majority_vote_from_dict(vote_counts)
                vote_distributions[dim] = vote_counts
        
        final_outputs.append({
            'text': original_example['text'],
            'comment_id': original_example['comment_id'],
            'annotator_id': llm_model_name,
            **majority_votes,
            'vote_distributions': json.dumps(vote_distributions),
            'rater_method': 'LLM_majority',
            'rater_type': 'LLM'
        })
    
    print("✓ Majority voting complete")
    
    return pd.DataFrame(final_outputs)

def get_llm_ratings_batching(
    dataset_samples,
    rater_dimension,
    n_runs,
    llm_model_name,
    batch_size=10
):
    """
    Process in batches to save intermediate results
    """
    
    num_samples = len(dataset_samples)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_results = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        batch = dataset_samples[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({len(batch)} samples)")
        
        batch_results = get_llm_ratings_with_majority_vote(
            batch,
            rater_dimension,
            n_runs,
            llm_model_name
        )
        
        all_results.append(batch_results)
        
        # Save checkpoint
        checkpoint_file = f'checkpoint_{llm_model_name}_batch_{batch_idx + 1}.csv'
        batch_results.to_csv(checkpoint_file, index=False)
        print(f"✓ Saved checkpoint: {checkpoint_file}")
    
    # Combine all batches
    final_df = pd.concat(all_results, ignore_index=True)
    
    return final_df
