"""Launch the AdaBoost s1 manual UI on loopback only."""

import os

import uvicorn

if __name__ == "__main__":
    os.environ["ERIS_ADABOOST_UI_HOST"] = "127.0.0.1"
    uvicorn.run(
        "eris_ml.manual_ui.adaboost_app:app",
        host="127.0.0.1",
        port=8511,
        access_log=False,
        log_level="warning",
    )
