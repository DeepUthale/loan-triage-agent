FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.12.5

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY app ./app
COPY config ./config
COPY .streamlit ./.streamlit

RUN useradd --create-home triage && mkdir -p /app/data && chown -R triage /app/data
USER triage

ENV LOAN_TRIAGE_DATA_DIR=/app/data
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["/app/.venv/bin/streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.headless=true"]
