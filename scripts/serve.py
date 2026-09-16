"""Run the API development server."""

import argparse


def main() -> None:
    """Run Uvicorn after explicit invocation."""
    parser = argparse.ArgumentParser(description="Serve the ERIS AI API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("eris_ml.api.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
