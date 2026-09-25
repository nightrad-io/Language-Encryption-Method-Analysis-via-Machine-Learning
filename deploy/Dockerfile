# syntax=docker/dockerfile:1
#
# Cipher/language-ID API + web UI, built straight from GitHub: nothing from a
# local checkout is copied in, so this directory is all a Docker host needs.
#
#   docker compose up -d --build                        # GIT_REF defaults to main
#   GIT_REF=expand-91-languages docker compose up -d --build
#
# GIT_REF must be a branch or tag (git clone --branch). Each fresh clone pulls
# ~720MB of models through Git LFS, which counts against the repository's LFS
# bandwidth quota.

FROM python:3.14-slim AS fetch
RUN apt-get update \
 && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG GITHUB_REPO=nightrad-io/Language-Encryption-Method-Analysis-via-Machine-Learning
ARG GIT_REF=main

# Cache key for the clone below: the ref's current commit metadata. Docker
# re-fetches this URL on every build and only invalidates later layers when its
# content changes -- a moved branch gets a fresh clone, an unchanged one reuses
# the cached models instead of pulling 720MB from LFS again.
ADD https://api.github.com/repos/${GITHUB_REPO}/commits/${GIT_REF} /tmp/ref.json

# The last check fails the build if any model is still an LFS pointer file
# (git lfs ls-files marks those with "-" instead of "*").
RUN git lfs install --skip-repo \
 && git clone --depth 1 --branch "${GIT_REF}" "https://github.com/${GITHUB_REPO}.git" /src \
 && git -C /src lfs pull \
 && if git -C /src lfs ls-files | grep -q ' - '; then echo "LFS objects missing" >&2; exit 1; fi

FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=fetch /src/requirements.txt /src/requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

COPY --from=fetch /src/models ./models
COPY --from=fetch /src/client.py ./
COPY --from=fetch /src/pipeline ./pipeline
COPY --from=fetch /src/server ./server

RUN useradd --system --uid 10001 app
USER app

EXPOSE 8000
# Loading the models takes a while; /api/health only answers once they're in.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]
# One worker: each worker process would hold its own copy of the models.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
