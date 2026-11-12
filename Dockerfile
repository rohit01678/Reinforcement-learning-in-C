FROM python:3.13-slim AS c-builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY *.c *.h ./

RUN gcc -std=c11 -Wall -Wextra -O3 \
    Snake.c arena.c random.c environment.c matrix.c model.c training.c \
    -lm -o Snake

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SNAKE_PREBUILT=1 \
    SNAKE_POLICY_PATH=/data/Snake-policy.bin

WORKDIR /app

RUN mkdir -p /data
VOLUME ["/data"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsdl2-2.0-0 \
        libsdl2-image-2.0-0 \
        libsdl2-mixer-2.0-0 \
        libsdl2-ttf-2.0-0 \
        libportmidi0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir pygame==2.6.1

COPY snake.py ./
COPY --from=c-builder /build/Snake ./Snake

ENTRYPOINT ["python", "snake.py", "--ai"]