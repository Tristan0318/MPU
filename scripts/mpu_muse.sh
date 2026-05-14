#!/usr/bin/env bash
set -euo pipefail

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
# Stage switches
############################################
RUN_FINETUNE_RETAIN="${RUN_FINETUNE_RETAIN:-1}"
RUN_EVAL_RETAIN="${RUN_EVAL_RETAIN:-1}"
RUN_FINETUNE_FULL="${RUN_FINETUNE_FULL:-1}"
RUN_EVAL_FULL="${RUN_EVAL_FULL:-1}"
RUN_UNLEARN="${RUN_UNLEARN:-1}"
RUN_EVAL_UNLEARN="${RUN_EVAL_UNLEARN:-1}"

############################################
# Models / Dataset
############################################
# 这里切到 Llama-3.2-1B-Instruct。
# OpenUnlearning 已支持该模型配置，但用于 MUSE 时属于你自己的实验设定，
# 不是官方 MUSE 默认 target model。
models=(
  "Llama-3.2-1B-Instruct"
)

data_splits=(
  "News"
  "Books"
)

# 这里保留你 MPU/PUM 的 client_method 风格。
# 对 MUSE 来说，experiment 应切到 unlearn/muse/default。
# DPO 需要额外的 idk 配置；OpenUnlearning 官方 MUSE baseline 脚本并未提供，先注释。
client_methods_experiments=(
  "GradAscent unlearn/muse/default.yaml"
  "GradDiff unlearn/muse/default.yaml"
  "NPO unlearn/muse/default.yaml"
  "SimNPO unlearn/muse/default.yaml"
  "UnDIAL unlearn/muse/default.yaml"
  "SatImp unlearn/muse/default.yaml"
  "WGA unlearn/muse/default.yaml"
  # "DPO unlearn/muse/idk.yaml"
)

############################################
# Dataset split settings for standard MUSE
############################################
# unlearn/muse/default.yaml 默认就是 forget / retain1，
# 这里显式写出来，避免后续改 config 时混淆。
MUSE_FORGET_SPLIT="forget"
MUSE_RETAIN_SPLIT="retain1"

############################################
# PUM hyperparameters
############################################
PUM_KAPPA=0.01
PUM_M_LIST=(2)
PUM_ETA_SRV=1.0
PUM_ROPE_AWARE=true
PUM_REPARAM=true

############################################
# Training parameters
############################################
CLIENT_LR=1e-5

# Finetune stage
FINETUNE_PER_DEVICE_TRAIN_BATCH_SIZE="${FINETUNE_PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
FINETUNE_GRADIENT_ACCUMULATION_STEPS="${FINETUNE_GRADIENT_ACCUMULATION_STEPS:-4}"
FINETUNE_NUM_TRAIN_EPOCHS="${FINETUNE_NUM_TRAIN_EPOCHS:-10}"

# Unlearn stage
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

# 关闭 train 内 eval；分布式 / 多卡下更稳，训练后单独 eval
EVAL_STRATEGY=no

# GPU 控制
FINETUNE_CUDA_VISIBLE_DEVICES="${FINETUNE_CUDA_VISIBLE_DEVICES:-0}"
UNLEARN_CUDA_VISIBLE_DEVICES="${UNLEARN_CUDA_VISIBLE_DEVICES:-0}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

# accelerate / deepspeed 配置
USE_ACCELERATE_FOR_FINETUNE="${USE_ACCELERATE_FOR_FINETUNE:-0}"
USE_ACCELERATE_FOR_UNLEARN="${USE_ACCELERATE_FOR_UNLEARN:-0}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate/default_config.yaml}"

############################################
# Helper
############################################
fmt_float_for_name() {
  echo "$1" | sed 's/\./p/g'
}

ensure_local_file() {
  local fpath="$1"
  local msg="$2"
  if [[ ! -e "${fpath}" ]]; then
    echo "[pum_muse.sh][ERROR] Missing: ${fpath}"
    echo "[pum_muse.sh][ERROR] ${msg}"
    exit 1
  fi
}

ensure_retain_logs() {
  local retain_logs_path="$1"
  if [[ -f "${retain_logs_path}" ]]; then
    return 0
  fi

  echo "[pum_muse.sh][ERROR] retain_logs_path not found: ${retain_logs_path}"
  echo "[pum_muse.sh][ERROR] Please run the RETAIN finetune + eval stage first, or set RUN_FINETUNE_RETAIN=1 RUN_EVAL_RETAIN=1"
  exit 1
}

run_train_cmd() {
  local gpu_ids="$1"
  shift
  if [[ "${USE_ACCELERATE_FOR_UNLEARN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_ids}" accelerate launch \
      --config_file "${ACCELERATE_CONFIG}" \
      --main_process_port "${MASTER_PORT}" \
      "$@"
  else
    CUDA_VISIBLE_DEVICES="${gpu_ids}" python "$@"
  fi
}

run_finetune_cmd() {
  local gpu_ids="$1"
  shift
  if [[ "${USE_ACCELERATE_FOR_FINETUNE}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_ids}" accelerate launch \
      --config_file "${ACCELERATE_CONFIG}" \
      --main_process_port "${MASTER_PORT}" \
      "$@"
  else
    CUDA_VISIBLE_DEVICES="${gpu_ids}" python "$@"
  fi
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
      echo "[pum_muse.sh][WARN] Unknown client_method='${method}', fallback to R=1 E=10" >&2
      echo "1 10"
      ;;
  esac
}

############################################
# Main loops
############################################
for model in "${models[@]}"; do
  case "${model}" in
    Llama-3.2-1B-Instruct)
      base_model_id="meta-llama/${model}"
      ;;
    Llama-2-7b-hf)
      base_model_id="meta-llama/${model}"
      ;;
    *)
      echo "[pum_muse.sh][WARN] Unknown base model mapping for ${model}; using model name directly as base_model_id"
      base_model_id="${model}"
      ;;
  esac

  for data_split in "${data_splits[@]}"; do
    retrain_task_name="muse_${model}_${data_split}_retrain"
    full_task_name="muse_${model}_${data_split}_full"

    retrain_model_dir="saves/finetune/${retrain_task_name}"
    full_model_dir="saves/finetune/${full_task_name}"
    retain_logs_path="saves/eval/${retrain_task_name}/MUSE_EVAL.json"

    echo "================================================================================"
    echo "[pum_muse.sh] model=${model} data_split=${data_split}"
    echo "[pum_muse.sh] base_model_id=${base_model_id}"
    echo "[pum_muse.sh] retrain_model_dir=${retrain_model_dir}"
    echo "[pum_muse.sh] full_model_dir=${full_model_dir}"
    echo "[pum_muse.sh] retain_logs_path=${retain_logs_path}"
    echo "================================================================================"

    # ------------------------------------------------------------------
    # Stage 1A: finetune RETAIN model on tamarsonha/MUSE-${data_split}-Train:retain
    # ------------------------------------------------------------------
    if [[ "${RUN_FINETUNE_RETAIN}" == "1" ]]; then
      echo "[pum_muse.sh] Finetuning RETAIN model: ${retrain_task_name}"
      run_finetune_cmd "${FINETUNE_CUDA_VISIBLE_DEVICES}" \
        src/train.py --config-name=train.yaml \
        experiment=finetune/muse/default.yaml \
        task_name="${retrain_task_name}" \
        model="${model}" \
        data_split="${data_split}" \
        data_sub_set=retain \
        trainer.args.eval_strategy="${EVAL_STRATEGY}" \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.per_device_train_batch_size="${FINETUNE_PER_DEVICE_TRAIN_BATCH_SIZE}" \
        trainer.args.gradient_accumulation_steps="${FINETUNE_GRADIENT_ACCUMULATION_STEPS}" \
        trainer.args.num_train_epochs="${FINETUNE_NUM_TRAIN_EPOCHS}" \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true
    fi

    # ------------------------------------------------------------------
    # Stage 1B: eval RETAIN model to produce retain_logs_path for MUSE privleak
    # ------------------------------------------------------------------
    if [[ "${RUN_EVAL_RETAIN}" == "1" ]]; then
      echo "[pum_muse.sh] Evaluating RETAIN model: ${retrain_task_name}"
      ensure_local_file "${retrain_model_dir}" "RETAIN model checkpoint is missing. Run RUN_FINETUNE_RETAIN=1 first."
      CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
      python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split="${data_split}" \
        model="${model}" \
        task_name="${retrain_task_name}" \
        model.model_args.pretrained_model_name_or_path="${retrain_model_dir}"
    fi

    # ------------------------------------------------------------------
    # Stage 2A: finetune FULL target model on tamarsonha/MUSE-${data_split}-Train:full
    # ------------------------------------------------------------------
    if [[ "${RUN_FINETUNE_FULL}" == "1" ]]; then
      echo "[pum_muse.sh] Finetuning FULL target model: ${full_task_name}"
      run_finetune_cmd "${FINETUNE_CUDA_VISIBLE_DEVICES}" \
        src/train.py --config-name=train.yaml \
        experiment=finetune/muse/default.yaml \
        task_name="${full_task_name}" \
        model="${model}" \
        data_split="${data_split}" \
        data_sub_set=full \
        trainer.args.eval_strategy="${EVAL_STRATEGY}" \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.per_device_train_batch_size="${FINETUNE_PER_DEVICE_TRAIN_BATCH_SIZE}" \
        trainer.args.gradient_accumulation_steps="${FINETUNE_GRADIENT_ACCUMULATION_STEPS}" \
        trainer.args.num_train_epochs="${FINETUNE_NUM_TRAIN_EPOCHS}" \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true
    fi

    # ------------------------------------------------------------------
    # Stage 2B: optional eval FULL target model (baseline / sanity check)
    # ------------------------------------------------------------------
    if [[ "${RUN_EVAL_FULL}" == "1" ]]; then
      echo "[pum_muse.sh] Evaluating FULL target model: ${full_task_name}"
      ensure_local_file "${full_model_dir}" "FULL model checkpoint is missing. Run RUN_FINETUNE_FULL=1 first."
      ensure_retain_logs "${retain_logs_path}"
      CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
      python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split="${data_split}" \
        model="${model}" \
        task_name="${full_task_name}" \
        model.model_args.pretrained_model_name_or_path="${full_model_dir}" \
        retain_logs_path="${retain_logs_path}"
    fi

    # ------------------------------------------------------------------
    # Stage 3: PUM unlearning on standard MUSE forget/retain1 split
    # ------------------------------------------------------------------
    if [[ "${RUN_UNLEARN}" == "1" ]]; then
      ensure_local_file "${full_model_dir}" "FULL target model checkpoint is missing. Run RUN_FINETUNE_FULL=1 first."
      ensure_retain_logs "${retain_logs_path}"

      for cm_exp in "${client_methods_experiments[@]}"; do
        client_method=$(echo "${cm_exp}" | cut -d' ' -f1)
        experiment=$(echo "${cm_exp}" | cut -d' ' -f2)
        experiment="${experiment%.yaml}"

        read -r PUM_R CLIENT_ROUND_EPOCH < <(get_round_epoch_pair_for_method "${client_method}")
        TOTAL_EPOCHS=$((PUM_R * CLIENT_ROUND_EPOCH))

        for PUM_M in "${PUM_M_LIST[@]}"; do
          kappa_tag="$(fmt_float_for_name "${PUM_KAPPA}")"
          task_name="pum_muse_${model}_${data_split}_${client_method}_k${kappa_tag}_m${PUM_M}_R${PUM_R}_E${CLIENT_ROUND_EPOCH}"
          unlearn_out_dir="saves/unlearn/${task_name}"
          eval_out_dir="${unlearn_out_dir}/evals"

          echo "================================================================================"
          echo "[pum_muse.sh] task_name=${task_name}"
          echo "[pum_muse.sh] model=${model}"
          echo "[pum_muse.sh] data_split=${data_split}"
          echo "[pum_muse.sh] forget_split=${MUSE_FORGET_SPLIT}, retain_split=${MUSE_RETAIN_SPLIT}"
          echo "[pum_muse.sh] client_method=${client_method} => R=${PUM_R}, E=${CLIENT_ROUND_EPOCH} (R*E=${TOTAL_EPOCHS})"
          echo "[pum_muse.sh] PUM: kappa=${PUM_KAPPA} m=${PUM_M} eta_srv=${PUM_ETA_SRV} rope_aware=${PUM_ROPE_AWARE} reparam=${PUM_REPARAM}"
          echo "[pum_muse.sh] Trainer: bs=${PER_DEVICE_TRAIN_BATCH_SIZE}, grad_acc=${GRADIENT_ACCUMULATION_STEPS}, num_train_epochs=${TOTAL_EPOCHS}"
          echo "[pum_muse.sh] experiment=${experiment}"
          echo "================================================================================"

          run_train_cmd "${UNLEARN_CUDA_VISIBLE_DEVICES}" \
            src/train.py --config-name=unlearn.yaml \
            experiment="${experiment}" \
            trainer=PUM \
            task_name="${task_name}" \
            model="${model}" \
            data_split="${data_split}" \
            forget_split="${MUSE_FORGET_SPLIT}" \
            retain_split="${MUSE_RETAIN_SPLIT}" \
            model.model_args.pretrained_model_name_or_path="${full_model_dir}" \
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

          if [[ "${RUN_EVAL_UNLEARN}" == "1" ]]; then
            mkdir -p "${eval_out_dir}"
            CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
            python src/eval.py \
              experiment=eval/muse/default.yaml \
              data_split="${data_split}" \
              model="${model}" \
              task_name="${task_name}" \
              model.model_args.pretrained_model_name_or_path="${unlearn_out_dir}" \
              paths.output_dir="${eval_out_dir}" \
              retain_logs_path="${retain_logs_path}"
            echo "[pum_muse.sh][DONE] Eval saved in ${eval_out_dir}"
          fi
        done
      done
    fi
  done
done
