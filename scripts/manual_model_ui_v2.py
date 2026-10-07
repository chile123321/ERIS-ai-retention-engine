"""Run the isolated V2 Streamlit UI on loopback with telemetry disabled."""

from pathlib import Path

from streamlit.web import cli

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cli.main(args=[
        "run", str(ROOT / "src/eris_ml/manual_ui/v2_streamlit.py"),
        "--server.address=127.0.0.1", "--server.port=8502", "--server.headless=true",
        "--server.fileWatcherType=none", "--server.runOnSave=false",
        "--server.enableStaticServing=false", "--server.enableCORS=true",
        "--server.enableXsrfProtection=true", "--browser.gatherUsageStats=false",
        "--browser.serverAddress=127.0.0.1", "--client.toolbarMode=minimal",
        "--client.showErrorDetails=none", "--theme.font=sans-serif",
        "--theme.headingFont=sans-serif", "--theme.codeFont=monospace",
    ], prog_name="streamlit")
