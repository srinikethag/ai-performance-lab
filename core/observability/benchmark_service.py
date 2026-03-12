from prometheus_client import start_http_server
import time

from benchmark_instrumented import run_benchmark


def main():

    start_http_server(8000)

    print("Metrics endpoint running at:")
    print("http://localhost:8000/metrics")

    while True:

        run_benchmark()

        time.sleep(5)


if __name__ == "__main__":
    main()