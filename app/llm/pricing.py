
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""
    input       : float
    output      : float
    cache_read  : float
    cache_write : float


# Claude API first-party rates
PRICES: dict[str, Price] = {
    'claude-haiku-5-5'  : Price(input=0.10, output=0.50,  cache_read=0.01, cache_write=0.125),
    'claude-sonnet-5-5' : Price(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
}

# Haiku 5.5 bills prompts above 100K input tokens at the long-context rate.
HAIKU_LONG_CONTEXT_TOKENS: int = 100_000
HAIKU_LONG_CONTEXT = Price(input=0.50, output=2.50, 
                           cache_read=0.05, cache_write=0.625)


def estimate_cost(model: str, 
                  input_tokens: int, 
                  output_tokens: int,
                  cache_read: int = 0, 
                  cache_write: int = 0
                  ) -> float | None:
    """Estimated USD for one call, or None for a model without a price."""
    price = PRICES.get(model)
    if price is None:
        return None

    prompt_tokens = input_tokens + cache_read + cache_write
    if model == 'claude-haiku-5-5' and prompt_tokens > HAIKU_LONG_CONTEXT_TOKENS:
        price = HAIKU_LONG_CONTEXT
    usd = (input_tokens * price.input + output_tokens * price.output
           + cache_read * price.cache_read + cache_write * price.cache_write)

    return usd / 1_000_000



