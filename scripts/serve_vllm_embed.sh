#!/usr/bin/env bash
# Start multilingual-e5-large as a vLLM OpenAI-compatible embeddings server.
#
#   bash agentic_rag_v2/scripts/serve_vllm_embed.sh            # foreground
#   bash agentic_rag_v2/scripts/serve_vllm_embed.sh --detach   # background + wait
#   bash agentic_rag_v2/scripts/serve_vllm_embed.sh --stop
#
# Uses the vllm/vllm-openai image that already lives in the WSL docker daemon and
# the weights downloaded to /path/to/models/multilingual-e5-large.
#
#   --runner pooling        serve an embedding model (not a generator)
#   --pooler-config mean    E5 is trained with mean pooling; vLLM defaults to CLS,
#                           which produces different (worse) vectors
#   model as positional     `vllm serve` deprecates --model
set -euo pipefail

NAME=${VLLM_NAME:-vllm-e5}
PORT=${VLLM_PORT:-18000}
MODELS_DIR=${VLLM_MODELS_DIR:-/path/to/models}
MODEL=${VLLM_MODEL:-$MODELS_DIR/multilingual-e5-large}
# the volume mounts MODELS_DIR at /models, so the container must be given the
# *container* path of the model, not the host path
case "$MODEL" in
  /models/*) CONTAINER_MODEL="$MODEL" ;;
  *) CONTAINER_MODEL="/models/$(basename "$MODEL")" ;;
esac
SERVED=${VLLM_SERVED_NAME:-multilingual-e5-large}
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:latest}
GPU_MEM=${VLLM_GPU_MEM:-0.25}
DETACH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --detach|-d) DETACH=1; shift ;;
    --stop) docker rm -f "$NAME" >/dev/null 2>&1 || true; echo "stopped $NAME"; exit 0 ;;
    --port) PORT=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

docker rm -f "$NAME" >/dev/null 2>&1 || true

if [ "$DETACH" = "1" ]; then
  docker run -d --name "$NAME" --gpus all --ipc=host \
     -v "$MODELS_DIR:/models" \
     -p "$PORT:8000" \
     -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
     "$IMAGE" \
     "$CONTAINER_MODEL" \
     --runner pooling \
     --pooler-config '{"pooling_type":"MEAN"}' \
     --served-model-name "$SERVED" \
     --max-model-len 512 \
     --dtype float16 \
     --gpu-memory-utilization "$GPU_MEM" >/dev/null
  echo "waiting for http://127.0.0.1:$PORT/v1/models ..."
  for _ in $(seq 1 150); do
    if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
      curl -s "http://127.0.0.1:$PORT/v1/models"; echo; exit 0
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
      echo "container exited; last logs:" >&2
      docker logs --tail 40 "$NAME" >&2 || true
      exit 1
    fi
    sleep 2
  done
  echo "TIMEOUT: docker logs $NAME" >&2
  docker logs --tail 40 "$NAME" >&2 || true
  exit 1
fi

exec docker run --name "$NAME" --gpus all --ipc=host \
   -v "$MODELS_DIR:/models" \
   -p "$PORT:8000" \
   -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
   "$IMAGE" \
   "$CONTAINER_MODEL" \
   --runner pooling \
   --pooler-config '{"pooling_type":"MEAN"}' \
   --served-model-name "$SERVED" \
   --max-model-len 512 \
   --dtype float16 \
   --gpu-memory-utilization "$GPU_MEM"


