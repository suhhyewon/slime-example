#!/bin/bash
# Local smoke-test launch script for Qwen3-4B multi-turn IF RL on RTX 2080 Ti
# Adapted from qwen3-14b-mt.sh. See deep-yawning-stallman.md plan for rationale.
# All paths assume:
#   - SLIME framework at SLIME_DIR (default /var/tmp/hsuh45-slime/yxli2123-slime)
#   - this slime-example/ at SLIME_EXAMPLE_DIR (auto-detected)
#   - scratch storage at SCRATCH_DIR (default /var/tmp/hsuh45-slime)
# Pre-reqs (do once, see install_dependency_local.sh + plan):
#   - conda env "slime" activated, with SLIME / Megatron-LM / sglang installed
#   - flite + nltk + spacy data installed
#   - HF datasets pre-downloaded into HF_HOME
#
# NOTE: soft constraints are disabled in this smoke config (the YAML below
# sets disable_soft_verifiers: true). No judge LLM is required. To re-enable
# later, set disable_soft_verifiers: false and configure judge_base_url +
# judge_api_key(_path).

set -ex

# ===================== Cleanup any stale processes =====================
pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
pkill -9 python3 || true
sleep 2

# ===================== Paths =====================
SLIME_DIR=${SLIME_DIR:-/var/tmp/hsuh45-slime/thudm-slime}
SCRATCH_DIR=${SCRATCH_DIR:-/var/tmp/hsuh45-slime}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

mkdir -p "${SCRATCH_DIR}"/{hf_cache,torch_cache,checkpoints,data,logs,configs,bin}

export HF_HOME="${SCRATCH_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH_DIR}/hf_cache"
export TORCH_HOME="${SCRATCH_DIR}/torch_cache"
export PATH="${SCRATCH_DIR}/bin:${PATH}"
export PYTHONUNBUFFERED=1               # was PYTHONBUFFERED=16 in original (typo)
# wandb: defaults to offline (smoke). To run online: `wandb login` once
# (writes to ~/.netrc), then start the script with `WANDB_MODE=online`. Project
# and run-name are overridable via env so smoke runs and real runs can share
# the script.
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-qwen3-mt-if-rl}
export WANDB_DIR=${WANDB_DIR:-${SCRATCH_DIR}/wandb}
mkdir -p "${WANDB_DIR}"

# Disable transformer_engine on Turing (sm_75 unsupported by TE).
export NVTE_TORCH_COMPILE=0
# Force PyTorch SDPA path before trying flash-attn 2.x kernels on Turing.
export NVTE_FUSED_ATTN=0
# Megatron's jit.py auto-uses torch.compile (inductor) on torch>=2.2 — that
# trips on Turing during backward. Disable dynamo entirely so jit_fuser falls
# back to torch.jit.script.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# ===================== Hardware probe =====================
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ===================== Model =====================
# Smoke-test note: we originally targeted Qwen3-4B but the 8 GB FP32 gradient
# buffer for 4B/TP=2 doesn't fit in 11 GB cards. Qwen3-1.7B exercises the
# exact same pipeline (same RMSNorm/qk-layernorm/GQA/swiglu/rotary) and fits
# easily. Re-enable 4B once we have larger VRAM.
source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"

HF_REPO_ID=${HF_REPO_ID_OVERRIDE:-Qwen/Qwen3-1.7B}

# Turing (sm_75) cannot install transformer_engine. Disable Megatron features
# that hard-require TE (rope fusion), force the non-TE layer spec at model
# build time, and pick an attention backend that doesn't need NVTE/flash-attn.
# These flags need to flow into both convert+train calls, so go in MODEL_ARGS
# (sourced into both commands). Megatron's --attention-backend choices are
# flash|fused|unfused|local|auto — `unfused` is the safe non-TE path.
MODEL_ARGS+=( --no-rope-fusion --transformer-impl local --attention-backend unfused --no-persist-layer-norm --fp16 )

HF_MODEL_PATH=${SCRATCH_DIR}/models/${HF_REPO_ID}
MEGATRON_MODEL_PATH=${SCRATCH_DIR}/megatron-models/${HF_REPO_ID}

if [ ! -d "${HF_MODEL_PATH}" ]; then
  hf download "${HF_REPO_ID}" --local-dir "${HF_MODEL_PATH}"
fi

if [ ! -d "${MEGATRON_MODEL_PATH}" ]; then
  # 11 GB cards can't hold a 4B FP16 model + CUDA context (~3 GB) on a single
  # GPU during the convert's get_model().cuda() call. Megatron only skips
  # that .cuda() under FSDP2, not just --use-cpu-initialization. Workaround:
  # shard the convert across 2 GPUs with TP=2. The torch_dist checkpoint
  # format is TP-agnostic at load time so the resulting checkpoint still
  # loads cleanly at training-time TP.
  PYTHONPATH=${SLIME_DIR}/Megatron-LM:${PYTHONPATH:-} \
  torchrun --nproc_per_node 2 --master_port 12399 \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --use-cpu-initialization \
    --hf-checkpoint "${HF_MODEL_PATH}" \
    --save "${MEGATRON_MODEL_PATH}"
  # NB: convert_hf_to_torch_dist auto-sets pipeline_model_parallel_size to
  # world_size (here 2) — do not pass --tensor-model-parallel-size or PP/CP
  # explicitly. The torch_dist checkpoint format is parallelism-agnostic at
  # load time so training can still use TP=2/PP=1.
fi

# ===================== Data =====================
# NB: the colleague's original 1220-train / 1207-eval datasets are no longer
# on HF Hub. For the smoke test we fall back to yxli2123/mtif-eval-2turn-500
# which has the same schema (id, prompt, verifier, persona). 500 rows is more
# than enough — the smoke run only does 2 rollouts of 8 prompts each.
HF_TRAIN_DATA=${HF_TRAIN_DATA:-yxli2123/mtif-eval-2turn-500}
HF_EVAL_DATA=${HF_EVAL_DATA:-yxli2123/mtif-eval-2turn-500}
LOCAL_TRAIN_DATA=${SCRATCH_DIR}/data/${HF_TRAIN_DATA}.jsonl
LOCAL_EVAL_DATA=${SCRATCH_DIR}/data/${HF_EVAL_DATA}.jsonl

if [ ! -f "${LOCAL_TRAIN_DATA}" ] || [ ! -f "${LOCAL_EVAL_DATA}" ]; then
  python "${SCRIPT_DIR}/convert_to_slime_data.py" \
    --hf-train-data "${HF_TRAIN_DATA}" \
    --hf-eval-data "${HF_EVAL_DATA}" \
    --local-train-data "${LOCAL_TRAIN_DATA}" \
    --local-eval-data "${LOCAL_EVAL_DATA}"
fi

# ===================== Custom-args YAML =====================
# Vanilla SLIME doesn't have --judge_* / --disable_soft_verifiers in argparse.
# We use --custom-config-path so SLIME setattr's these onto args at parse time.
# `multi_if_reward.py` reads:
#   args.disable_soft_verifiers, args.verify_helpfulness_rate,
#   args.judge_api_key_path, args.judge_api_key, args.judge_base_url
#
# Soft constraints are OFF for the smoke test — no judge endpoint is needed.
# To re-enable: flip disable_soft_verifiers to false, set judge_base_url to a
# reachable OpenAI-compatible endpoint, and put a real key in API_KEY_PATH.
CUSTOM_CFG=${SCRATCH_DIR}/configs/custom_args.yaml
cat > "${CUSTOM_CFG}" <<EOF
disable_soft_verifiers: true
verify_helpfulness_rate: 0.0
judge_api_key_path: null
judge_api_key: null
judge_base_url: null
EOF

# Pre-cache the constraint pool and our HF data so workers don't re-download.
# Constraint pool dataset is overridable via env var (we added the indirection
# in multi_if_reward.py).
export CONSTRAINT_POOL_NAME=${CONSTRAINT_POOL_NAME:-yxli2123/verifiable-constraints-1126}

# ===================== Experiment configs =====================
PROJ_NAME=${PROJ_NAME:-qwen3-4b-mt-smoke}
EXPT_NAME=${EXPT_NAME:-smoke-$(date +%Y%m%d-%H%M%S)}

CKPT_ROOT=${CKPT_ROOT:-${SCRATCH_DIR}/checkpoints}
CKPT_DIR=${CKPT_ROOT}/${PROJ_NAME}/${EXPT_NAME}
CKPT_ARGS=(
   --hf-checkpoint "${HF_MODEL_PATH}"
   --ref-load "${MEGATRON_MODEL_PATH}"
   --load "${CKPT_DIR}"
   --save "${CKPT_DIR}"
   --save-interval 1
)

ROLLOUT_ARGS=(
   --prompt-data "${LOCAL_TRAIN_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT:-6}       # smoke: enough iterations to see FP16 grad scaler stabilize
   --rollout-batch-size 8                # was 64
   --n-samples-per-prompt 4              # was 8
   --rollout-max-response-len 1024       # was 6144
   --rollout-temperature 0.6
   --global-batch-size 32                # = rollout-batch-size * n-samples-per-prompt
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 1
   --eval-prompt-data "${LOCAL_EVAL_DATA}"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-p 0.8
)

# Turing-specific: TP=2, no TE, FP16.
PERF_ARGS=(
   # TP=4 with 8 actor GPUs (DP=2) keeps params + grad buffer + sharded Adam
   # state at ~5 GB per rank, leaving plenty of headroom for the rollout-1
   # update_weights staging buffer that OOM'd at TP=2.
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # Shard optimizer state across DP ranks (ZeRO-1) — without this Adam state
   # for 4B/TP=2 alone is 16 GB, well over 11 GB VRAM.
   --use-distributed-optimizer

   # --use-dynamic-batch-size packs sequences and requires TEDotProductAttention.
   # On Turing without TE we use Megatron's DotProductAttention which doesn't
   # support packed seqs, so leave dynamic batching off and force bshd
   # (batched + padded) layout instead of thd (packed).
   --qkv-format bshd
   --max-tokens-per-gpu 2048             # was 8192; 11 GB cards
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# Note: original used BF16 (default in MODEL_ARGS for many model scripts).
# Turing has no hardware BF16 — force FP16 explicitly.
PRECISION_ARGS=(
   # --fp16 lives in MODEL_ARGS (so the convert step also uses FP16, otherwise
   # FP32 weights of a 4B model don't fit in 11 GB during .cuda() inside the
   # conversion's get_model call).
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-tensor-parallel-size 1
   --sglang-mem-fraction-static 0.7      # was 0.8; smaller GPUs need a bit more headroom
   # sgl-kernel ships only sm_80+ binaries — force pure-PyTorch sampling and
   # the triton attention backend on Turing (sm_75). SLIME prefixes sglang
   # ServerArgs flags with --sglang-, so these become --sampling-backend etc.
   --sglang-sampling-backend pytorch
   --sglang-attention-backend triton
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # --attention-backend is set in MODEL_ARGS above so that the convert step
   # also picks it up (Megatron asserts on it before any flag-only override).
)

# wandb + richer logging. The rollout/* and train/* metrics are emitted
# automatically; these flags add multi-turn breakdown, pass@n, reward-category
# stats, and a sample dump of correct rollouts.
LOGGING_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${PROJ_NAME}"
   --log-multi-turn
   --log-passrate
   # --log-reward-category requires reward_func to return a dict (with the
   # given key); multi_if_reward returns a float, so leave this off until we
   # rework reward_func to emit per-category breakdown.
   # --log-correct-samples has an off-by-one in slime where it indexes a
   # per-rank-shard array with the global-batch index; triggers IndexError
   # whenever any sample has raw_reward==1.
)

CUSTOM_ARGS=(
   --custom-generate-function-path multi_if_generate.generate
   --custom-rm-path multi_if_reward.reward_func
   --custom-config-path "${CUSTOM_CFG}"
)

# ===================== Ray =====================
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
TOTAL_GPUS=$(nvidia-smi -L | wc -l)
ray start --head \
   --node-ip-address "${MASTER_ADDR}" \
   --num-gpus "${TOTAL_GPUS}" \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_DIR}/Megatron-LM/:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_TORCH_COMPILE\": \"0\",
    \"NVTE_FUSED_ATTN\": \"0\",
    \"TORCHDYNAMO_DISABLE\": \"1\",
    \"TORCH_COMPILE_DISABLE\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"CONSTRAINT_POOL_NAME\": \"${CONSTRAINT_POOL_NAME}\",
    \"WANDB_MODE\": \"${WANDB_MODE}\",
    \"WANDB_PROJECT\": \"${WANDB_PROJECT}\",
    \"WANDB_DIR\": \"${WANDB_DIR}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"HF_HOME\": \"${HF_HOME}\",
    \"TRANSFORMERS_CACHE\": \"${TRANSFORMERS_CACHE}\"
  }
}"

# 2080 Ti × 9 split: 8 actor + 1 rollout. Actor uses TP=4/DP=2 for headroom;
# 1.7B FP16 (3.4 GB weights) fits in a single 11 GB sglang engine alongside
# its KV cache. Adjust if fewer GPUs are free.
ACTOR_GPUS=${ACTOR_GPUS:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}

cd "${SLIME_DIR}"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${PRECISION_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${LOGGING_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
