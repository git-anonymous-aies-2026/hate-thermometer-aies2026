from datetime import datetime
from pathlib import Path
import os, sys, math
import RaschPy as rp
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import config
print(f"Config imported successfully!")

from config import DATA_DIR, OUTPUT_DIR, DIMENSIONS, COLORS, LLM_NAMES, ANNOTATOR_ID_MAP

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def read_filepaths(base_dir=DATA_DIR):
    if not base_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {base_dir}")
    file_map = {}
    for f in base_dir.glob("*.csv"):
        name = f.name.lower()
        if "gold" in name:
            file_map["gold"] = f
        elif "gpt5_mono" in name:
            file_map["GPT5_mono"] = f

        elif "gpt-5" in name and "mini_mono" in name:
            file_map["gpt_5-mini_mono"] = f
        elif "gpt-5" in name and "mini_modu" in name:
            file_map["gpt_5-mini_modu"] = f

        elif "claude" in name and "mono" in name:
            file_map["claude_mono"] = f
        elif "claude" in name and "modu" in name:
            file_map["claude_modu"] = f

        elif "gemini" in name and "mono" in name:
            file_map["gemini_mono"] = f
        elif "gemini" in name and "modu" in name:
            file_map["gemini_modu"] = f

        elif "deepseek" in name and "mono" in name:
            file_map["deepseek_mono"] = f
        elif "deepseek" in name and "modu" in name:
            file_map["deepseek_modu"] = f

        elif "llm_human_mod_mono" in name:  #filtering the 4 extrement human annotations 
            file_map["llm_human_mod_mono"] = f

        elif "unfil_llm_human_mod_mono" in name:
            file_map["unfil_llm_human_mod_mono"] = f
        elif "llm_only_mod_mono" in name:
            file_map["llm_only_mod_mono"] = f
        elif "llm_only_mono" in name:
            file_map["llm_only_mono"] = f
        elif 'llm_mono_nogpt5' in name:
            file_map['llm_mono_nogpt5'] = f
        elif 'llm_only_modu' in name:
            file_map['llm_only_modu'] = f
    return file_map

def filter_extreme_raters(df, dimensions=DIMENSIONS[:-1], 
                           min_variance=0.01):
    """
    Remove raters whose ratings have zero or near-zero variance
    across all dimensions — these cause RaschPy fit statistic errors.
    """
    rater_variances = df.groupby('annotator_id')[dimensions].var().mean(axis=1)
    extreme_raters  = rater_variances[rater_variances < min_variance].index.tolist()
    
    if extreme_raters:
        print(f"⚠️ Removing {len(extreme_raters)} extreme raters: {extreme_raters}")
        df = df[~df['annotator_id'].isin(extreme_raters)].copy()
    else:
        print("✓ No extreme raters found")
    
    return df

def data_differentiate_raters_3(df):
    def get_llm_label(rater_id):
        rater_id    = str(rater_id).strip()
        rater_lower = rater_id.lower() #.split('_')[0]  

        # Human — keep numeric ID as-is
        if rater_id.isnumeric():
            return rater_id

        # Step 1 — check explicit map first (handles non-standard IDs like yours)
        if rater_lower in ANNOTATOR_ID_MAP:
            return ANNOTATOR_ID_MAP[rater_lower]

        MODELS       = ['GPT5', 'gpt-5-mini', 'claude', 'gemini'] #'deepseek'
        PROMPT_TYPES = ['mono', 'modu']

        # Step 2 — generalised string detection (handles well-named future models)
        detected_model = next((m for m in MODELS if m in rater_lower), None)
        if detected_model is None:
            return rater_id  # unknown — return as-is

        detected_prompt = next((p for p in PROMPT_TYPES if p in rater_lower), None)
        return f'{detected_model}_{detected_prompt}' if detected_prompt else detected_model

    rater_params = df.copy()
    rater_params['annotator_id'] = rater_params['annotator_id'].apply(get_llm_label)
    return rater_params

def creating_mfrm_ready_data(filepath_gold, 
                            filepath_gpt5_mono, 
                            filepath_gpt5mini_mono, 
                            filepath_gpt5mini_modu,
                            filepath_claude_mono, 
                            filepath_claude_modu,
                            filepath_gemini_mono,
                            filepath_gemini_modu, 
                            filepath_deepseek_mono,
                            filepath_deepseek_modu,
                            output_dir):  

    os.makedirs(output_dir, exist_ok=True)

    dataset_dict = {
        'gold': filepath_gold, 
        'GPT5_mono': filepath_gpt5_mono, 
        'gpt-5-mini_mono': filepath_gpt5mini_mono, 
        'gpt-5-mini_modu': filepath_gpt5mini_modu,
        'claude_mono': filepath_claude_mono, 
        'claude_modu': filepath_claude_modu,
        'gemini_mono': filepath_gemini_mono,
        'gemini_modu': filepath_gemini_modu, 
        'deepseek_mono': filepath_deepseek_mono,
        'deepseek_modu': filepath_deepseek_modu
    }

    pd_datasets = {}
    mfrm_datasets = {}
    all_dataframes = {}

    for dataset_name, filepath in dataset_dict.items():
        pd_datasets[dataset_name] = pd.read_csv(filepath)
        mfrm_datasets[dataset_name] = pd_datasets[dataset_name][
            ['annotator_id', 'comment_id'] + DIMENSIONS].copy()
        all_dataframes[dataset_name] = data_differentiate_raters_3(
            mfrm_datasets[dataset_name])


    all_dataframes['llm_human_mod_mono'] = pd.concat(
        [all_dataframes[k] for k in [ 'gold', 'GPT5_mono', 'gpt-5-mini_mono',
                                    'gpt-5-mini_modu', 'claude_mono', 'claude_modu',
                                    'gemini_mono', 'gemini_modu', 'deepseek_mono', 'deepseek_modu']], #deepseek_mono, deepseek_modu
        ignore_index=True)

    #llm_mono with gpt5
    all_dataframes['llm_only_mono'] = pd.concat([
        all_dataframes[k] for k in [ 'GPT5_mono', 'gpt-5-mini_mono',
                                     'claude_mono','gemini_mono', 'deepseek_mono']], #deepseek_mono
        ignore_index=True)

    #llm_mono without gpt5
    all_dataframes['llm_mono_nogpt5'] = pd.concat([
        all_dataframes[k] for k in [ 'gpt-5-mini_mono',
                                     'claude_mono','gemini_mono', 'deepseek_mono']], #deepseek_mono
        ignore_index=True)

    #llm_modu only
    all_dataframes['llm_only_modu'] = pd.concat([
        all_dataframes[k] for k in [ 'gpt-5-mini_modu',
                                     'claude_modu','gemini_modu', 'deepseek_modu']], #deepseek_mono
        ignore_index=True)

    #llm_mono and llm_modu only
    all_dataframes['llm_only_mod_mono'] = pd.concat([
        all_dataframes[k] for k in [ 'GPT5_mono', 'gpt-5-mini_mono',
                                    'gpt-5-mini_modu', 'claude_mono', 'claude_modu',
                                    'gemini_mono', 'gemini_modu', 'deepseek_mono', 'deepseek_modu']], #deepseek_mono, deepseek_modu
        ignore_index=True)


    for dataset_name, df in all_dataframes.items():
        mfrm_save = f'mfrm_{dataset_name}_ready_{timestamp}.csv'
        mfrm_csv_file = os.path.join(output_dir, mfrm_save)
        df.to_csv(mfrm_csv_file, index=False)
        # print(f"✓ Saved {dataset_name} MFRM-ready data to: {mfrm_csv_file}") 

    all_data_concat = all_dataframes['llm_human_mod_mono']
    print(f"\nCombined dataset:")
    print(f"Total rows in concatenated filtered df: {len(all_data_concat)}")
    print(f"Unique comments: {all_data_concat['comment_id'].nunique()}")
    print(f"Unique raters: {all_data_concat['annotator_id'].nunique()}")
    # print(f"\nRater breakdown:") 
    print("\nchecking for missing values")

    # Check for missing values
    missing_summary = all_data_concat[DIMENSIONS].isnull().sum()
    if missing_summary.sum() > 0:
        print("\n[Problem] WARNING: Missing values detected:")
        print(missing_summary[missing_summary > 0])
    else:
        print("\n[GOOD] No missing values in dimension ratings")

    return pd_datasets, all_dataframes, all_data_concat

def interpret_strictest_llm(row, llm_cols=LLM_NAMES):
    """
    Find strictest LLM (highest difficulty = most conservative)
    """
    max_val   = row[llm_cols].max()
    strictest = [col for col in llm_cols if row[col] == max_val]
    return strictest[0] if len(strictest) == 1 else "Tie: " + " & ".join(strictest)

def interpret_most_lenient_llm(row,llm_cols=LLM_NAMES):
    """
    Find most lenient LLM (lowest difficulty = most liberal)
    """
    min_val   = row[llm_cols].min()
    lenient = [col for col in llm_cols if row[col] == min_val]
    return lenient[0] if len(lenient) == 1 else "Tie: " + " & ".join(lenient)


def get_severity(severity_df):
    rater_params = severity_df.copy()
    def get_llm_label(rater_id):
        rater_id    = str(rater_id)
        rater_lower = rater_id.lower()
        if rater_id.isdigit():
            return 'Human'
        elif 'gpt5_mono' in rater_lower:
            model = 'GPT5_mono'
        elif 'gpt-5-mini' in rater_lower:
            model = 'gpt-5-mini'
        elif 'claude' in rater_lower:
            model = 'claude'
        elif 'gemini' in rater_lower:
            model = 'gemini'
        elif 'deepseek' in rater_lower:
            model = 'deepseek'
        else:
            return 'GPT5_mono'

        #detect prompting strategy
        if 'mono' in rater_lower:
            return f'{model}_mono'
        elif 'modu' in rater_lower:
            return f'{model}_modu'
        else:
            return model

    rater_params= rater_params.reset_index()
    print(rater_params.columns)
    rater_params['rater_type'] = rater_params['Rater'].map(get_llm_label)
    
    # Reset index to make rater_id a column
    rater_params.rename(columns={'Rater':'annotator', 'Estimate': 'severity'}, inplace=True)
    rater_params_plotting = rater_params[['annotator', 'severity', 'rater_type']]
    
    return rater_params, rater_params_plotting

def plot_severity_strip(rater_params, OUTPUT_DIR=None, colors=COLORS):
    """
    Plot showing severity distribution per annotator type.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define group order
    groups = ['Human', 'GPT5_mono', 'gpt-5-mini_mono', 'gpt-5-mini_modu', 
            'claude_mono', 'claude_modu', 'gemini_mono', 'gemini_modu', 
            'deepseek_mono', 'deepseek_modu'] #

    groups = [g for g in groups if g in rater_params['rater_type'].unique()]

    for i, group in enumerate(groups):
        subset = rater_params[rater_params['rater_type'] == group]['severity']
        color  = colors.get(group, '#888888')
        n      = len(subset)

        # Jitter x positions so points don't overlap
        jitter = np.random.uniform(-0.15, 0.15, size=n)
        x_pos  = i + jitter

        # Plot individual points
        ax.scatter(
            x_pos, subset,
            color=color, alpha=0.3 if group == 'Human' else 1,
            s=30 if group == 'Human' else 50,
            zorder=2,
            label=f"{group} (n={n})"
        )

        # Mean line
        ax.plot(
            [i - 0.25, i + 0.25], [subset.mean(), subset.mean()],
            color=color, lw=2.5, zorder=4
        )

        # 95% CI bar for humans
        if group == 'Human' and n > 10:
            ci = stats.sem(subset) * 1.96
            ax.errorbar(
                i, subset.mean(),
                yerr=ci,
                fmt='none',
                color=color, lw=1.5,
                capsize=4, zorder=5
            )

        # Annotate LLM exact values (since n=1 or very small)
        if group != 'Human':
            for val in subset:
                ax.annotate(
                    f"{val:.2f}",
                    xy=(i + 0.3, val),
                    fontsize=8.5, color=color, va='center'
                )

    # Reference line at 0 (average severity)
    ax.axhline(0, color='black', lw=1, linestyle='--', alpha=0.4, label='Mean severity (0)')

    # # Shade the "lenient" and "harsh" regions
    y_max = rater_params['severity'].max() + 0.5
    y_min = rater_params['severity'].min() - 0.5
    ax.set_ylim(y_min, y_max)
    ax.axhspan(0.5,  y_max, alpha=0.04, color='red',  label='Harsh region (>+0.5)')
    ax.axhspan(y_min, -0.5, alpha=0.04, color='blue', label='Lenient region (<-0.5)')

    # Formatting
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=6)
    ax.set_ylabel("Severity (logits)", fontsize=13)
    ax.set_xlabel("Annotator Type", fontsize=13)
    ax.set_title(
        "Rater Severity Distribution: Humans vs LLMs\n"
        "Horizontal bar = group mean  |  Positive = harsh, Negative = lenient",
        fontsize=15
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if OUTPUT_DIR:
        plt.savefig(OUTPUT_DIR, dpi=150, bbox_inches='tight')

    plt.show()


def plot_severity_with_context_zscore(rater_params, OUTPUT_DIR=None, colors=COLORS):
    """
    Where do LLMs fall within the human severity distribution?
    """
    # fig, ax = plt.subplots(figsize=(10, 5))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    humans = rater_params[rater_params['rater_type'] == 'Human']['severity']
    human_mean = humans.mean()
    human_std  = humans.std()

    llms   = rater_params[rater_params['rater_type'] != 'Human']

    #plotting a 2 in 1, first the logit, then the plot with the SD]
    #logit plot    # Human distribution as histogram
    ax1 = axes[0]
    ax1.hist(humans, bins=30, color=colors['Human'], alpha=0.5,
            label=f'Human raters (n={len(humans)})', edgecolor='white')

    # LLM vertical lines
    for _, row in llms.iterrows():
        color = colors.get(row['rater_type'], '#888')
        ax1.axvline(
            row['severity'],
            color=color, lw=2.5, linestyle='--',
            label=f"{row['annotator']} ({row['severity']:.2f})"
        )
    ax1.axvline(human_mean, color='black', lw=1.5, linestyle=':',
               label=f'Human mean ({human_mean:.2f})')
    ax1.set_xlabel("Severity (logits)", fontsize=15)
    ax1.set_ylabel("Number of human raters", fontsize=15)
    ax1.set_title(
        "Severity logit scale\n"
        "Dashed lines = LLM severity | Histogram = human raters",
        fontsize=15
    )
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    #SD plots
    ax2 = axes[1]
    humans_z = (humans - human_mean) / human_std #human z score
    ax2.hist(humans_z, bins=30, color=colors['Human'], alpha=0.5,
             label=f'Human raters (n={len(humans)})', edgecolor='white')
    #LLM severity z-score
    for _, row in llms.iterrows():
        color   = colors.get(row['rater_type'], '#888')
        z_score = (row['severity'] - human_mean) / human_std
        # what % of human raters are more lenient than this LLM
        from scipy import stats as scipy_stats
        percentile = scipy_stats.norm.cdf(z_score) * 100
        
        ax2.axvline(z_score, 
                    color=color, lw=2.5, linestyle='--',
                    label=f"{row['annotator']} "
                          f"(z={z_score:.2f} | P%={percentile:.0f}th)")

    ax2.axvline(0, color='black', lw=1.5, linestyle=':',
                label='Human mean (z=0)')

    ax2.set_xlabel("Severity (standard deviations from human mean)", fontsize=15)
    ax2.set_ylabel("Number of human raters", fontsize=15)
    ax2.set_title("Z-Score Scale\n Dashed lines = LLM severity | Histogram = human raters",
                  fontsize=15)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # plt.suptitle(
    #     "Where LLMs Fall Within the Human Severity Distribution",
    #     fontsize=14, fontweight='bold', y=1.02
    # )

    plt.tight_layout()

    if OUTPUT_DIR:
        plt.savefig(OUTPUT_DIR, dpi=150, bbox_inches='tight')

    plt.show()

def plot_human_vs_llm_scatter_theta(all_theta_df, output_dir=None):
    
    llm_cols = [c for c in all_theta_df.columns 
                if c not in ('human', 'joint')]
    
    n_cols = 3
    n_rows = math.ceil(len(llm_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(12, n_rows * 3))
    axes = axes.flatten()

    for i, name in enumerate(llm_cols):
        ax  = axes[i]
        x   = all_theta_df['human']
        y   = all_theta_df[name]
        
        r, p   = stats.pearsonr(x, y)
        mae    = (x - y).abs().mean()
        bias   = (y - x).mean()
        color  = COLORS.get(name, '#888888')

        ax.scatter(x, y, alpha=0.5, s=30, color=color)

        # Perfect agreement line
        lims = [min(x.min(), y.min()) - 0.5,
                max(x.max(), y.max()) + 0.5]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='Perfect agreement')

        # Regression line
        m, b_reg = np.polyfit(x, y, 1)
        ax.plot(lims, [m * l + b_reg for l in lims],
                color=color, lw=1.5, label=f'Fit (slope={m:.2f})')

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Human θ', fontsize=13)
        ax.set_ylabel(f'{name} θ', fontsize=13)
        ax.set_title(
            f'{name}\nr={r:.3f}  MAE={mae:.3f}  Bias={bias:+.3f}',
            fontsize=10, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        'Human vs LLM Continuous Hate Scores (θ)\n'
        'Each dot = one comment | Dashed = perfect agreement',
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir, dpi=150, bbox_inches='tight')
    plt.show()


def extract_theta(mfrm, label=""):
    mfrm.person_stats_df_global()
    theta = mfrm.person_stats_global.get('Estimate')
    theta.index.name = 'comment_id'
    if theta is None or theta.empty:
        raise ValueError(f"No theta estimates found for: {label}")
    return theta

def calibrate(filepath, label):
    """Helper to load and calibrate MFRM model."""
    print(f"  Calibrating {label}...")
    data, _ = rp.loadup_mfrm_single(filepath)
    model   = rp.MFRM(data)
    model.calibrate_global()
    return model