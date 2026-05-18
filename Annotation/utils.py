# ===========================================================================================
# PRELIMINARIES: PUTTING THE DATA IN THE RIGHT FORMAT
# ============================================================================================
import re, os
import ast
import json
import pandas as pd
import dspy
import numpy as np


def get_majority_vote_from_dict(vote_dict):
    if not vote_dict or sum(vote_dict.values()) == 0:
        return None
    max_count = max(vote_dict.values())
    top_votes = [int(v) for v, c in vote_dict.items() if c == max_count]  
    
    if len(top_votes) == 1:
        return top_votes[0]   
    return int(np.median(top_votes))  

def safe_int_convert(value, default=None):
    """Safely convert to int, handling None and invalid values."""
    if value is None or pd.isna(value):
        return default
    # Handle string cases like '{5}' or '5'
    if isinstance(value, str):
        value = value.strip('{}').strip()
    try:
        return int(float(value))  # float first handles decimals
    except (ValueError, TypeError):
        return default


def convert_labels_to_numeric(label):
    if label is None:
        return None
    mapping = {
        'yes'    : 2,
        'unclear': 1,
        'no'     : 0
    }
    cleaned = str(label).strip().lower().rstrip('.')
    result  = mapping.get(cleaned, None)
    if result is None:
        for key, val in mapping.items():
            if key in cleaned:
                return val
    return result


def convert_to_dspy_example(row):
    """
    Convert row to DSPy Example for LLM inference
    No gold labels - LLM will generate predictions
    """
    return dspy.Example(
        comment_id=row['comment_id'],
        text=row['text'],
        slur_tagged_text=row['slur_tagged_text']
    ).with_inputs('comment_id', 'slur_tagged_text')

def create_safe_pattern(slur):
    escaped = re.escape(slur)
    pattern = rf"(?<!['\w])\b{escaped}\b(?!['\w])"
    return re.compile(pattern, re.IGNORECASE)

def load_slur_database(): 
    current_dir = os.path.dirname(__file__)   
    filepath = os.path.join(current_dir, 'rsdb_slurs_with_dups.json')
    with open(filepath, 'r') as file:
        slur_database = json.load(file)
    # Sort by length (longest first) to prevent partial matching
    slur_database_sorted = sorted(
        slur_database, 
        key=lambda x: len(x['slur ']), 
        reverse=True)

    return slur_database_sorted

def apply_slur_tags_placeholder(text, slur_db):    
    # Tag (using placeholder method), replace all slurs with some placeholder and theur re substitue tag in he end
    tagged_text = text
    for i, entry in enumerate(slur_db):
        slur = entry['slur '].strip()
        pattern = create_safe_pattern(slur)
        
        if pattern.search(tagged_text):
            placeholder = f"___SLUR{i}___"
            tagged_text = pattern.sub(placeholder, tagged_text)
    
    # Replace placeholders with real tags
    for i, entry in enumerate(slur_db):
        placeholder = f"___SLUR{i}___"
        if placeholder in tagged_text:
            demo_group = entry['target'].strip()
            explanation = entry['explanation'].strip()
            slur = entry['slur '].strip()
            tag = f"<slur demo_group='{demo_group}' explanation='{explanation}'>{entry['slur ']}</slur>"
            tagged_text = tagged_text.replace(placeholder, tag)
    return tagged_text

def extract_incomplete_rows(df, n_runs):
    incomplete_indices = []
    for row in df.itertuples():
        try:
            vote_dist = json.loads(row.vote_distributions)
            for dim, votes in vote_dist.items():
                if votes and sum(votes.values()) < n_runs:
                    incomplete_indices.append(row.Index)
                    break
        except:
            incomplete_indices.append(row.Index)
    return df.loc[incomplete_indices].copy(), incomplete_indices