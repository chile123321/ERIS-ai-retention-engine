FROM python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 eris \
    && useradd --uid 10001 --gid eris --no-create-home --shell /usr/sbin/nologin eris
COPY deploy/api_b2/requirements-runtime.txt /app/requirements-runtime.txt
RUN python -m pip install --no-compile --only-binary=:all: -r /app/requirements-runtime.txt \
    && python -m pip check

# .dockerignore permits only the existing inference import closure, not all src.
COPY src/eris_ml/ /app/src/eris_ml/
COPY artifacts/models/v3_final_15/b2/ /app/artifacts/models/v3_final_15/b2/
COPY artifacts/models/adaboost_v3/s1/ /app/artifacts/models/adaboost_v3/s1/
RUN chmod -R a-w /app && chmod -R a+rX /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=4 \
    CMD ["python", "-B", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).close()"]
CMD ["python", "-B", "-m", "uvicorn", "eris_ml.api_b2.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
