
from pathlib import Path
from tokenize import Single
from pydantic_settings import BaseSettings, SettingsConfigDict
 

# Single source of truth for runtime configuration. All paths, thresholds and
# toggles come from environment variables or a `.env` file at the repo root.
 
 
class Settings(BaseSettings):
        
    # Pipeline artifacts (paths relative to repo root by default)
        
    onnx_path        : Path = Path('artifacts/best_model_v9.onnx')
    class_map        : Path = Path('artifacts/class_map50_v9.json')
    trie_pkl         : Path = Path('artifacts/sphinx_trie_v4.pkl')
    bbaw_parquet     : Path = Path('artifacts/bbaw_clean.parquet')
    confusion_csv    : Path = Path('artifacts/confusion_matrix_v9_normalized.csv')
 
    
    # YOLO / ONNX inference parameters

    imgsz            : int   = 1024     # must match training imgsz
    conf_threshold   : float = 0.25     # min confidence to keep a detection
    iou_threshold    : float = 0.45     # NMS IoU threshold
    max_detections   : int   = 300      # cap per image after NMS
 

    # Pipeline parameters
    default_dir      : str   = 'rtl'    # 'rtl' or 'ltr'
    default_preset   : str   = 'default'
 

    # Request limits
 
    max_upload_mb    : int   = 20       # hard cap on uploaded image size
    request_timeout  : float = 30.0     # seconds before pipeline run aborts
 
 
    # CORS — comma-separated origins in env, parsed to list
 
    cors_origins     : list[str] = ['*']
 
    # LLM transliteration (Phase 6)
    translit_max_signs  : int   = 40     # max Gardiner codes per LLM chunk
                                         #   (raised from 20 2026-09-08: longer
                                         #    segments = more grammar per call)

    # Anthropic (app/llm/). anthropic_api_key reads ANTHROPIC_API_KEY; empty = disabled.
    # No temperature: current Claude models reject non-default sampling values.
    anthropic_api_key   : str   = ''
    haiku_model         : str   = 'claude-haiku-5-5'
    sonnet_model        : str   = 'claude-sonnet-5-5'
    chat_effort         : str   = 'low'
    translit_effort     : str   = 'medium'
    reverse_effort      : str   = 'medium'
    llm_timeout         : float = 120.0  # seconds per request, SDK retries on top

    # Debug: app/llm/ dumps each request's system prompt, messages and reply
    # to debug_prompts/ (SPHINX_DEBUG_PROMPTS=1 in .env). Payload only, never
    # the API key; image data is redacted. Read per call, so no restart needed.
    debug_prompts       : bool  = False

    # Metadata

    pipeline_version : str   = 'v9'
 
    model_config = SettingsConfigDict(
        env_file        = '.env',
        env_file_encoding = 'utf-8',
        case_sensitive  = False,
        extra           = 'ignore',
    )
 
 
# Single instance imported everywhere
settings = Settings()
