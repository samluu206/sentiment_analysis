"""Script to start MLflow tracking server UI."""
import subprocess
import sys
import argparse
from pathlib import Path


def start_mlflow_ui(port: int = 5000, host: str = "127.0.0.1"):
    """
    Start MLflow UI server.

    Args:
        port: Port to run the server on
        host: Host address to bind to
    """
    project_root = Path(__file__).parent.parent
    mlruns_path = project_root / "mlruns"

    mlruns_path.mkdir(exist_ok=True)

    print(f"Starting MLflow UI server...")
    print(f"  Backend store: {mlruns_path}")
    print(f"  Server address: http://{host}:{port}")
    print(f"\nAccess the UI at: http://localhost:{port}")
    print("\nPress Ctrl+C to stop the server")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                f"file://{mlruns_path}",
                "--host",
                host,
                "--port",
                str(port)
            ],
            check=True
        )
    except KeyboardInterrupt:
        print("\nStopping MLflow server...")
    except subprocess.CalledProcessError as e:
        print(f"Error starting MLflow server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start MLflow tracking server UI")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)"
    )

    args = parser.parse_args()
    start_mlflow_ui(port=args.port, host=args.host)
