from prometheus_client import Histogram, Counter, Gauge


# Time to first token
llm_ttft_seconds = Histogram(
    "llm_ttft_seconds",
    "Time to first token latency",
    buckets=(0.1,0.2,0.5,1,2,5,10)
)

# Total latency
llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "End to end LLM request latency",
    buckets=(0.5,1,2,5,10,20)
)

# Token counter
llm_tokens_generated_total = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated"
)

# Throughput
llm_tokens_per_second = Gauge(
    "llm_tokens_per_second",
    "Token generation throughput"
)