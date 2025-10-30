FROM python:3.10-slim 

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/ 

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock . 

RUN uv sync --locked 

COPY . /app 

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]  