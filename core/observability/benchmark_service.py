from prometheus_client import start_http_server
import time

from benchmark_instrumented import run_benchmark, MODELS


def main():

    start_http_server(8000)

    print("Metrics server running at http://localhost:8000/metrics")

    while True:

        for model in MODELS:

            print(f"Running benchmark for {model}")

            run_benchmark(model)

        time.sleep(2)


if __name__ == "__main__":
    main()