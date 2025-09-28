FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base
LABEL authors="duke_trystan"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl build-essential git

FROM base AS deps

COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /uvx /bin/
COPY pyproject.toml .
COPY .python-version .
COPY uv.lock .
RUN uv sync --frozen

FROM deps AS code
COPY db.py .
COPY bot_funcs.py .
COPY bot.py .

CMD ["uv", "run", "python", "bot.py"]