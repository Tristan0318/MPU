#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com

export MASTER_PORT=$(
  python - <<'PY'
import socket
s=socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)
echo "Master Port: $MASTER_PORT"

mkdir -p "${HOME}/.triton/autotune" || true
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

############################################
# Models / Experiments
############################################
models=(
  "Llama-3.2-1B-Instruct"
)

# client_method + experiment config
client_methods_experiments=(
  # "GradAscent unlearn/tofu/default.yaml"
  # "GradDiff   unlearn/tofu/default.yaml"
  "NPO unlearn/tofu/default.yaml"
  # "SimNPO     unlearn/tofu/default.yaml"
  # "UnDIAL     unlearn/tofu/default.yaml"
  # "SatImp     unlearn/tofu/default.yaml"
  # "WGA        unlearn/tofu/default.yaml"
  # "DPO        unlearn/tofu/idk.yaml"
)

splits=(
  "forget01 holdout01 retain99"
  # "forget05 holdout05 retain95"
  # "forget10 holdout10 retain90"
)

############################################
# PUM hyperparameters
############################################
PUM_KAPPA=0.01
PUM_M_LIST=(2)

PUM_ETA_SRV=1.0
PUM_ROPE_AWARE=true

PUM_REPARAM=true

############################################
# Client / Trainer parameters
############################################
CLIENT_LR=1e-5

PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4

# eval 控制
EVAL_STRATEGY=no

UNLEARN_CUDA_VISIBLE_DEVICES="${UNLEARN_CUDA_VISIBLE_DEVICES:-0}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

############################################
# Helper: ensure retain logs exist
############################################
SETUP_EVAL_DONE=0
ensure_retain_logs() {
  local retain_logs_path="$1"

  if [[ -f "${retain_logs_path}" ]]; then
    return 0
  fi

  echo "[pum_tofu.sh][WARN] retain_logs_path not found: ${retain_logs_path}"

  if [[ "${SETUP_EVAL_DONE}" -eq 0 ]]; then
    echo "[pum_tofu.sh][INFO] Running: python setup_data.py --eval"
    python setup_data.py --eval
    SETUP_EVAL_DONE=1
  fi

  if [[ ! -f "${retain_logs_path}" ]]; then
    echo "[pum_tofu.sh][ERROR] retain_logs_path still missing: ${retain_logs_path}"
    exit 1
  fi
}

fmt_float_for_name() {
  echo "$1" | sed 's/\./p/g'
}

############################################
# Core: choose (R, E) by method
############################################
get_round_epoch_pair_for_method() {
  local method="$1"
  case "$method" in
    # SimNPO / DPO / SatImp => R=10, E=1
    SimNPO|DPO|SatImp|Satlmp)
      echo "10 1"
      ;;

    # GradAscent / GradDiff / UnDIAL => R=1, E=10
    GradAscent|GradDiff|UnDIAL|UNDIAL)
      echo "1 10"
      ;;

    # NPO / WGA => R=2, E=5
    NPO|WGA)
      echo "2 5"
      ;;

    *)
      echo "[pum_tofu.sh][WARN] Unknown client_method='${method}', fallback to R=1 E=10" >&2
      echo "1 10"
      ;;
  esac
}

############################################
# Main loops
############################################
for split in "${splits[@]}"; do
  forget_split=$(echo "${split}" | cut -d' ' -f1)
  holdout_split=$(echo "${split}" | cut -d' ' -f2)
  retain_split=$(echo "${split}" | cut -d' ' -f3)

  for model in "${models[@]}"; do
    target_model_id="open-unlearning/tofu_${model}_full"
    base_model_id="meta-llama/${model}"

    retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
    ensure_retain_logs "${retain_logs_path}"

    for cm_exp in "${client_methods_experiments[@]}"; do
      client_method=$(echo "${cm_exp}" | cut -d' ' -f1)
      experiment=$(echo "${cm_exp}" | cut -d' ' -f2)

      experiment="${experiment%.yaml}"

      read -r PUM_R CLIENT_ROUND_EPOCH < <(get_round_epoch_pair_for_method "${client_method}")

      TOTAL_EPOCHS=$((PUM_R * CLIENT_ROUND_EPOCH))

      for PUM_M in "${PUM_M_LIST[@]}"; do
        kappa_tag="$(fmt_float_for_name "${PUM_KAPPA}")"

        task_name="pum_tofu_${model}_${forget_split}_${client_method}_k${kappa_tag}_m${PUM_M}_R${PUM_R}_E${CLIENT_ROUND_EPOCH}"

        unlearn_out_dir="saves/unlearn/${task_name}"
        eval_out_dir="${unlearn_out_dir}/evals"

        echo "===================================================================================================="
        echo "[pum_tofu.sh] task_name=${task_name}"
        echo "[pum_tofu.sh] model=${model}"
        echo "[pum_tofu.sh] split: forget=${forget_split}, retain=${retain_split}"
        echo "[pum_tofu.sh] client_method=${client_method} => R=${PUM_R}, E=${CLIENT_ROUND_EPOCH} (R*E=${TOTAL_EPOCHS})"
        echo "[pum_tofu.sh] PUM: kappa=${PUM_KAPPA} m=${PUM_M} eta_srv=${PUM_ETA_SRV} rope_aware=${PUM_ROPE_AWARE}"
        echo "[pum_tofu.sh] Trainer: bs=${PER_DEVICE_TRAIN_BATCH_SIZE}, grad_acc=${GRADIENT_ACCUMULATION_STEPS}, num_train_epochs=${TOTAL_EPOCHS}"
        echo "[pum_tofu.sh] experiment=${experiment}"
        echo "===================================================================================================="

        # -----------------------------
        # Unlearn (PUM)
        # -----------------------------
        CUDA_VISIBLE_DEVICES="${UNLEARN_CUDA_VISIBLE_DEVICES}" \
        python src/train.py --config-name=unlearn.yaml \
          experiment="${experiment}" \
          trainer=PUM \
          task_name="${task_name}" \
          model="${model}" \
          forget_split="${forget_split}" \
          retain_split="${retain_split}" \
          model.model_args.pretrained_model_name_or_path="${target_model_id}" \
          trainer.args.eval_strategy="${EVAL_STRATEGY}" \
          trainer.args.do_eval=false \
          trainer.args.eval_on_start=false \
          trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
          trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
          trainer.args.num_train_epochs="${TOTAL_EPOCHS}" \
          +trainer.pum.base_model_name_or_path="${base_model_id}" \
          trainer.pum.R="${PUM_R}" \
          trainer.pum.m="${PUM_M}" \
          trainer.pum.kappa="${PUM_KAPPA}" \
          trainer.pum.eta_srv="${PUM_ETA_SRV}" \
          trainer.pum.rope_aware="${PUM_ROPE_AWARE}" \
          +trainer.pum.reparam="${PUM_REPARAM}" \
          trainer.pum.client_method="${client_method}" \
          trainer.pum.client_round_epoch="${CLIENT_ROUND_EPOCH}" \
          trainer.pum.client_lr="${CLIENT_LR}" \
          retain_logs_path="${retain_logs_path}"

        # -----------------------------
        # Eval (TOFU)
        # -----------------------------
        mkdir -p "${eval_out_dir}"

        CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
        python src/eval.py \
          experiment=eval/tofu/default \
          forget_split="${forget_split}" \
          holdout_split="${holdout_split}" \
          model="${model}" \
          task_name="${task_name}" \
          model.model_args.pretrained_model_name_or_path="${unlearn_out_dir}" \
          paths.output_dir="${eval_out_dir}" \
          retain_logs_path="${retain_logs_path}"

        echo "[pum_tofu.sh][DONE] Eval saved in ${eval_out_dir}"
      done
    done
  done
done