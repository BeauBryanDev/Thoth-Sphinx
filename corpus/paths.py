import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('EVIDENCE_DATA_DIR',
                               Path.home() / 'Downloads' / 'AncientLanguage')).expanduser()
BUILD_DIR = DATA_DIR / 'build'
REPORTS_DIR = CODE_ROOT / 'reports'
KNOWLEDGE_DIR = CODE_ROOT / 'knowledge'
ENV_FILE = CODE_ROOT / '.env'
VISION_SIGNS = CODE_ROOT / 'corpus' / 'vision_signs.txt'
ROYAL_NAMES_FILE = CODE_ROOT / 'vision' / 'royal_names.json'
