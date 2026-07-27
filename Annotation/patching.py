"""
Functions for patching incomplete LLM annotation runs
"""

import ast
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import dspy
import json, os, sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import config, utils
from utils import apply_slur_tags_placeholder, convert_labels_to_numeric, convert_to_dspy_example, load_slur_database, get_majority_vote_from_dict
from config import DIMENSIONS

def normalise_vote_dict(d):
    return {str(k): v for k, v in d.items()} if d else {}

def patch_incomplete_runs(df, llm_model_name, rater_dimension, n_runs=5, max_attempts_per_missing=2):
    

    INVALID_VALUES = {'', 'none', 'null', 'n/a'}
    # Load slur database
    slur_database_sorted = load_slur_database()
    updated_df = df.copy()
    
    vote_dist_cache = {}
    rows_to_patch = []
    parse_errors = []

    print("Parsing vote distributions...")
    for row in tqdm(df.itertuples(), total=len(df), desc="Parsing"):
        try:
            vote_dist = json.loads(row.vote_distributions) # Convert string to dict
            vote_dist_cache[row.Index] = {
                dim: normalise_vote_dict(v)
                for dim, v in vote_dist.items()
            }
        except Exception as e:
            parse_errors.append((row.Index, str(e)))
            vote_dist_cache[row.Index] = None
    
    if parse_errors:
        print(f"\n⚠️ {len(parse_errors)} parsing errors:")
        for idx, error in parse_errors[:5]:
            print(f"  Row {idx+1}: {error}")

    # Identify rows needing patching
    for idx, vote_dist in vote_dist_cache.items():
        if vote_dist is None:continue
        needs_patching = False
        missing_by_dim = {
            dim: n_runs - sum(votes.values())
            for dim, votes in vote_dist.items()
            if votes and sum(votes.values()) < n_runs
        }

        if missing_by_dim:
            row_data = df.iloc[idx]
            rows_to_patch.append({
                'index': idx,     # Store index for later update, maybe +1 or +2 will be back to check later
                'comment_id': row_data['comment_id'],
                'text': row_data['text'],
                'slur_tagged_text': apply_slur_tags_placeholder(
                    row_data['text'], 
                    slur_database_sorted
                    ),
                'missing_by_dim': missing_by_dim,
                'max_missing': max(missing_by_dim.values()),
                'vote_dist' : vote_dist
            })
    
    if len(rows_to_patch) == 0:
        print("[Great!] No rows need patching here!")
        return df
    
    print(f"\nFound {len(rows_to_patch)} rows to patch")
    
    # Patch each row
    updates_by_index = {} # Track updates for batch application
    for patch_info in tqdm(rows_to_patch, desc="Patching rows"):
        
        idx = patch_info['index']
        comment_id = patch_info['comment_id']
        text = patch_info['text']
        slur_tagged_text = patch_info['slur_tagged_text']
        max_missing = patch_info['max_missing']
        existing_vote_dist = patch_info['vote_dist']
        
        print(f"\n{'='*80}") 
        print(f"Patching row {idx+1} (comment {comment_id}): need {max_missing} more runs") #will check on the row index later
        print(f"Text preview: {text[:150]}...")
        print(f"{'='*80}")
        
        # existing_vote_dist = ast.literal_eval(updated_df.loc[idx, 'vote_distributions'])
        new_votes_by_dim = {dim: [] for dim in DIMENSIONS}
        successful_runs = 0
        attempt = 0
        max_attempts = max_missing * max_attempts_per_missing
        
        while successful_runs < max_missing and attempt < max_attempts:
            attempt += 1
            try:
                dim_result = rater_dimension(text=text, slur_tagged_text=slur_tagged_text)
                # dspy.inspect_history(n=1) 
                # hate_result = rater_hate(text=text)
                # dspy.inspect_history(n=1)

                # Extract all dimension attributes at once
                dim_ratings = {
                    dim: getattr(dim_result, dim, None) 
                    for dim in DIMENSIONS[:-1]
                }
                
                all_valid = True

                for dim in DIMENSIONS[:-1]:
                    raw_value = dim_ratings[dim]
                    
                    if raw_value is None:
                        new_votes_by_dim[dim].append(None)
                        continue
                    
                    # Validate non-None values
                    try:
                        rating = int(raw_value)
                        if 0 <= rating <= 4:
                            new_votes_by_dim[dim].append(rating)
                        else:
                            all_valid = False
                            new_votes_by_dim[dim].append(None) #not sure if this is a good idea, will come baclk to check later
                    except:
                        all_valid = False
                        new_votes_by_dim[dim].append(None) #not sure if this is a good idea, will come baclk to check later

                raw_hate    = getattr(dim_result, 'hatespeech', None)
                numeric     = convert_labels_to_numeric(raw_hate)
                hate_rating = int(numeric) if numeric is not None else None
                new_votes_by_dim['hatespeech'].append(hate_rating)
                
                if all_valid:
                    successful_runs += 1
                    print(f"    ✓ Run {successful_runs}/{max_missing} complete")
                else:
                    print(f"    ⚠️ Run {attempt + 1} had invalid ratings, not counted")
                
            except Exception as e:
                print(f"    ⚠️ Attempt {attempt + 1} failed: {type(e).__name__}")
                time.sleep(0.5)
        
        print(f"  Completed {successful_runs}/{attempt} attempts")
        
        # Merge votes
        for dim in DIMENSIONS:
            current_votes = existing_vote_dist.get(dim, {})
            if not current_votes and all(v is None for v in new_votes_by_dim[dim]):
                continue
            current_total = sum(current_votes.values()) if current_votes else 0
            needed = n_runs - current_total
            if needed <= 0:
                continue
            new_votes_for_dim = [
                v for v in new_votes_by_dim[dim] 
                if v is not None
                ][:needed]
            for vote in new_votes_for_dim:
                vote = str(vote)
                current_votes[vote] = current_votes.get(vote, 0) + 1
            existing_vote_dist[dim] = current_votes
        
        # Recalculate majority
        row_updates = {
            dim: get_majority_vote_from_dict(existing_vote_dist.get(dim, {}))
            for dim in DIMENSIONS
        }
        row_updates['vote_distributions'] = json.dumps(existing_vote_dist) # Convert back to string for storage
        # Store updates for batch application
        updates_by_index[idx] = row_updates

    print("\nApplying updates in  bulk...")
    for idx, updates in tqdm(updates_by_index.items(), desc="Updating"):
        for col, value in updates.items():
            updated_df.loc[idx, col] = value
    print("✓ Patching complete")    
    return updated_df

def validate_completeness(df, n_runs=5):
    """
    Validate all rows have complete vote distributions
    Non-targeted comments will have empty vote dists for ordinal dimensions
    this is expected. Only flag dimensions that have SOME votes but fewer
    than n_runs.
    """

    vote_dists = []
    parse_errors = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Parsing"):
        try:
            # vote_dist = ast.literal_eval(row.vote_distributions)
            vote_dist = json.loads(row.vote_distributions)
            vote_dists.append((row.Index, vote_dist)) #tuple for eay access 
        except Exception as e:
            parse_errors.append((row.Index, str(e)))
            vote_dists.append((row.Index, None))
    
    if parse_errors:
        print(f"\n{len(parse_errors)} parsing errors found!")
        for idx, error in parse_errors[:10]:
            print(f"  Row {idx+1}: {error}")
        return False

    incomplete_issues = []
    for idx, vote_dist in tqdm(vote_dists, desc="Validating"):
        if vote_dist is None:  # Empty dict {}
            continue

        for dim in DIMENSIONS:
            votes = vote_dist.get(dim, {})
            if not votes: continue  # legitimately empty — non-targeted comment
            total_votes = sum(votes.values())
            if total_votes < n_runs:
                incomplete_issues.append({
                    'index': idx,
                    'dimension': dim,
                    'votes': total_votes,
                    'needed': n_runs,
                    'distribution': votes
                })

    if incomplete_issues:
        issues_by_index = {}
        for issue in incomplete_issues:
            idx = issue['index']
            if idx not in issues_by_index:
                issues_by_index[idx] = []
            issues_by_index[idx].append(issue)
        # print(f"\nAffected rows: {len(issues_by_index)}")
        print(f"\nVALIDATION FAILED: {len(incomplete_issues)} incomplete dimensions "
              f"across {len(issues_by_index)} rows")
        # Show first 5 rows with issues
        for idx in list(issues_by_index.keys())[:5]:
            print(f"\n  Row {idx+1}:")
            for issue in issues_by_index[idx]:
                print(f"    {issue['dimension']}: {issue['votes']}/{issue['needed']} votes")
        return False
    else:
        print(f"\nVALIDATION PASSED: All {len(df)} rows complete!")
        return True

