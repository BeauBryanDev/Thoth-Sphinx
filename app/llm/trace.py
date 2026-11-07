import time
from dataclasses import dataclass, field
from typing import Any, Literal

TraceKind = Literal['detection', 'llm_call', 'tool_call', 'router', 'escalation',
                    'evidence_check', 'final', 'error']


@dataclass
class Trace:
    """Append-only, ordered events of one read (spec §11). Never holds keys or image data."""
    started : float = field(default_factory=time.monotonic)
    events  : list[dict[str, Any]] = field(default_factory=list)

    def add(self, kind: TraceKind, 
            actor: str, summary: str, *,
            latency_ms: int | None = None, 
            input_tokens: int | None = None,
            output_tokens: int | None = None, 
            cost_usd: float | None = None,
            detail: dict[str, Any] | None = None
            ) -> dict[str, Any]:
        """Record one event and return it."""
        event = {
            'step'          : len(self.events) + 1,
            't_ms'          : int((time.monotonic() - self.started) * 1000),
            'kind'          : kind,
            'actor'         : actor,
            'summary'       : summary,
            'latency_ms'    : latency_ms,
            'input_tokens'  : input_tokens,
            'output_tokens' : output_tokens,
            'cost_usd'      : cost_usd,
            'detail'        : detail or {},
        }
        self.events.append(event)
        
        return event


    def add_usage(self, usage: Any, 
    summary: str, 
    detail: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Record one model turn from an app.llm.Usage."""
        return self.add('llm_call', usage.served_by, summary, 
                        latency_ms=usage.latency_ms,
                        input_tokens=usage.input_tokens + usage.cache_read + usage.cache_write,
                        output_tokens=usage.output_tokens, 
                        cost_usd=usage.cost_usd,
                        detail={
                            'feature': usage.feature,
                            'cache_read': usage.cache_read,
                            'cache_write': usage.cache_write,
                            'uncached_input': usage.input_tokens,
                            'fallback_used': usage.fallback_used,
                            **(detail or {}),
                        })
