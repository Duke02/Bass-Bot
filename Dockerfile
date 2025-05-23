FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL authors="duke_trystan"

WORKDIR /app

RUN apt-get update
RUN apt-get install -y curl build-essential git

COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /uvx /bin/
COPY pyproject.toml .
COPY .python-version .
COPY uv.lock .
ENV HF_HOME=/app/data/hf
RUN uv sync --frozen

COPY .env .
COPY db.py .
COPY bot_funcs.py .
COPY bot.py .

CMD ["uv", "run", "python", "bot.py"]