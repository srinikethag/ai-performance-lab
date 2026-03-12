from prometheus_client import Histogram, Counter, Gauge

llm_ttft_seconds = Histogram(
    "llm_ttft_seconds",
    "Time to first token latency",
    ["model"],
    buckets=(0.1,0.2,0.5,1,2,5,10)
)

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "End to end LLM request latency",
    ["model"],
    buckets=(0.5,1,2,5,10,20)
)

llm_tokens_generated_total = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated",
    ["model"]
)

llm_tokens_per_second = Gauge(
    "llm_tokens_per_second",
    "Token generation throughput",
    ["model"]
)