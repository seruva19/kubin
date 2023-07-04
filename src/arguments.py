import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Kubin")
    parser.add_argument("--from-config", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--flash-attention", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--share", type=str, default=None)
    parser.add_argument("--server-name", type=str, default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--concurrency-count", type=int, default=None)
    parser.add_argument("--debug", type=str, default=None)
    parser.add_argument("--extensions-path", type=str, default=None)
    parser.add_argument("--enabled-extensions", type=str, default=None)
    parser.add_argument("--disabled-extensions", type=str, default=None)
    parser.add_argument(
        "--extensions-order",
        type=str,
        default=None,
    )
    parser.add_argument("--skip-install", type=str, default=None)
    parser.add_argument("--safe-mode", type=str, default=None)
    parser.add_argument("--pipeline", type=str, default=None)
    parser.add_argument("--mock", type=str, default=None)
    parser.add_argument("--theme", type=str, default=None)
    parser.add_argument("--optimize", type=str, default=None)

    args = parser.parse_args()
    args_preview = {
        key: value for key, value in vars(args).items() if value is not None
    }
    print(f"command line arguments: {args_preview}")

    return args
