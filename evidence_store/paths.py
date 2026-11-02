"""Where the evidence store finds its code-side files and its data.

Code, reports, knowledge sources and .env live in this repo. The raw corpus, the vocabulary and
the build/ CSVs stay beside the database in EVIDENCE_DATA_DIR, which defaults to
~/Downloads/AncientLanguage.
"""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('EVIDENCE_DATA_DIR',
                               Path.home() / 'Downloads' / 'AncientLanguage')).expanduser()
BUILD_DIR = DATA_DIR / 'build'
REPORTS_DIR = CODE_ROOT / 'reports'
KNOWLEDGE_DIR = CODE_ROOT / 'knowledge'
ENV_FILE = CODE_ROOT / '.env'
VISION_SIGNS = CODE_ROOT / 'evidence_store' / 'vision_signs.txt'
ROYAL_NAMES_FILE = CODE_ROOT / 'royal_names.json'
