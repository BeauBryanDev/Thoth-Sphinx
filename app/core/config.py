
from pathlib import Path
from tokenize import Single
from pydantic import Field
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
    translit_whole_max_signs : int = 120 # up to this many codes: one call reads the whole wall
    translit_max_signs  : int   = 40     # max Gardiner codes per LLM chunk
                                         #   (raised from 20 2026-09-08: longer
                                         #    segments = more grammar per call)

    # Anthropic (app/llm/). anthropic_api_key reads ANTHROPIC_API_KEY; empty = disabled.
    # No temperature: current Claude models reject non-default sampling values.
    anthropic_api_key   : str   = ''
    haiku_model         : str   = 'claude-haiku-5-5'
    sonnet_model        : str   = 'claude-sonnet-5-5'
    translit_model      : str   = 'claude-haiku-5-5'   # first reader it was best LLM for this task
    # Escalation to a Sonnet second opinion by deterministically calling the router. Thresholds are
    # provisional hand-set defaults until Phase 3 calibration replaces them.
    translit_escalation : bool  = True
    review_model        : str   = 'claude-sonnet-5-5'
    review_effort       : str   = 'medium'
    router_max_low_line_share     : float = 0.5
    router_max_detector_low_share : float = 0.35
    # Corpus tools for the whole-wall read (app/llm/agent_tools.py); off = no tool loop
    translit_tools      : bool  = True
    # Tool budgets count TURNS (model round trips); calls within a turn run in parallel.
    haiku_tool_budget   : int   = 3      # was 5 calls
    sonnet_tool_budget  : int   = 2      # was 3 calls
    chat_effort         : str   = 'low'
    chat_tool_budget    : int   = 2        # /chat/: lookup_entity turns
    # Riddle trials in /chat/ (app/services/riddle_sessions.py): in memory, per process.
    riddle_session_ttl_s     : int = 3600   # an idle game is dropped after this
    riddle_max_sessions      : int = 2000   # oldest games are dropped beyond this
    # Cartouche gift (app/services/cartouche_image.py), unlocked by a passed riddle trial: the name is
    # carved on assets/cartouche_frame.webp; the OpenAI image settings make frames (scripts/generate_cartouche_frame.py).
    cartouche_model          : str   = 'gpt-image-1.5'   # gpt-image-1 retired by OpenAI 2026-10-23
    cartouche_size           : str   = '1024x1536'   # vertical
    cartouche_quality        : str   = 'low'
    cartouche_compression    : int   = 80            # WebP
    cartouche_timeout_s      : float = 120.0
    cartouche_images_per_day : int   = Field(3, ge=1)   # per client IP per UTC day; always on, never 0
    reading_chat_effort : str   = 'low'    # /chat/reading/: Haiku discussing the reading on screen
    reading_chat_tool_budget : int = 3
    reading_chat_history     : int = 12    # messages of the conversation sent each turn
    # Apply & re-read (a /transliterate/ call with human_corrections): once per client IP per
    # window. 0 disables the limit (local testing). In-memory, per process.
    reread_cooldown_s        : int = 3600
    # Guardrails (app/core/guards.py), in memory and per process like the re-read limit.
    # 0 disables each one (local testing).
    llm_requests_per_window  : int   = 20     # any LLM route, per client IP...
    llm_window_s             : int   = 600    # ...per 10 minutes
    translit_runs_per_day    : int   = 5      # transliterations (re-reads included) per client IP per UTC day
    daily_budget_usd         : float = 3.0    # whole agent, all clients; checked before every LLM call
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


