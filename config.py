# config.py
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent # relative to project root

# Then navigate to your data
DATA_DIR_RAW   = PROJECT_ROOT / 'data_outputs_ablate' / 'data' / 'raw_data'
DATA_DIR   = PROJECT_ROOT / 'data_outputs_ablate' / 'data' / 'patched_data'
DATA_DIR_MFRM   = PROJECT_ROOT / 'data_outputs_ablate' / 'data' / 'mfrm_data'
OUTPUT_DIR = PROJECT_ROOT / 'data_outputs_ablate' / 'outputs' / 'mfrm_results'
OUTPUT_DIR_stats = PROJECT_ROOT / 'data_outputs_ablate' / 'outputs' / 'stats'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_stats.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR_RAW.mkdir(parents=True, exist_ok=True)
DATA_DIR_MFRM.mkdir(parents=True, exist_ok=True)

# print(PROJECT_ROOT)


# ── Dimensions ────────────────────────────────────────────────────────────────
DIMENSIONS = [
    'sentiment', 'respect', 'insult', 'humiliate', 'status',
    'dehumanize', 'violence', 'genocide', 'attack_defend', 'hatespeech'
]

ORDINAL_DIMS = DIMENSIONS[:-1]
BINARY_DIM   = 'hatespeech'

# ── Colours ───────────────────────────────────────────────────────────────────

COLORS = {
    'Human':           '#85929E',   # stone gray
    'GPT5_mono':       '#E69F00',  # deep crimson'#C0392B'
    'gpt-5-mini_mono': '#E74C3C',   # bright red
    'gpt-5-mini_modu': '#922B21',   # dark blood red
    'claude_mono':     '#1A5276',   # deep navy blue
    'claude_modu':     '#56B4E9',   # strong cobalt blue #2980B9
    'gemini_mono':     '#27AE60',   # strong emerald green
    'gemini_modu':     '#1E8449',   # deep forest green
    'deepseek_mono':   '#6C3483',   # deep violet
    'deepseek_modu':   '#9B59B6',   # strong purple
}

# ── LLM annotator names ───────────────────────────────────────────────────────
LLM_NAMES = ['GPT5_mono', 'gpt-5-mini_mono', 'gpt-5-mini_modu', 'claude_mono', 'claude_modu', 
             'gemini_mono', 'gemini_modu', 'deepseek_mono', 'deepseek_modu']


ANNOTATOR_ID_MAP = {
    'GPT5_mono'              : 'GPT5_mono',
    'gpt-5-mini_mono'        : 'gpt-5-mini_mono',
    'gpt-5-mini_modu'        : 'gpt-5-mini_modu',
    'claude-haiku-4.5_mono'  : 'claude_mono',
    'claude-haiku-4.5_modu'  : 'claude_modu',
    'gemini-2.5-flash_mono'  : 'gemini_mono',
    'gemini-2.5-flash_modu'  : 'gemini_modu',
    'deepseek-v4-flash_mono' : 'deepseek_mono',
    'deepseek-v4-flash_modu' : 'deepseek_modu',
}
