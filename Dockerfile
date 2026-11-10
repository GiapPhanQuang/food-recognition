FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# 1) Dependencies only: this layer stays cached until uv.lock changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Our code: changes often, so it goes last
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" MODEL_PATH=/app/models/resnet18_food10.pt
EXPOSE 8000
CMD ["uvicorn", "food_recognition.api:app", "--host", "0.0.0.0", "--port", "8000"]
