export BS=${1:-16}
export MEMCAP=${2:-0}
export MODEL=${3:-"1.3b"}
export GPUNUM=${4:-1}
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# make directory for logs
mkdir -p ./logs

# env PYTORCH_NO_CUDA_MEMORY_CACHING=1
torchrun \
  --nproc_per_node ${GPUNUM} \
  --master_port 19198 \
  run_clm.py \
  --dataset_name AlekseyKorshuk/amazon-reviews-input-output \
  --model_name_or_path facebook/opt-${MODEL} \
  --output_dir $PWD \
  --mem_cap ${MEMCAP} \
  --per_device_train_batch_size ${BS} 2>&1 | tee ./logs/colo_${MODEL}_bs_${BS}_cap_${MEMCAP}_gpu_${GPUNUM}.log \
  --logging_strategy steps \
  --logging_steps 1 \
  --evaluation_strategy steps \
  --eval_steps 1 \
  --logging_first_step \
  --report_to all \
  --overwrite_output_dir \
  --num_train_epochs 5 \
  --fp16 \
  --push_to_hub \
  --hub_model_id "AlekseyKorshuk/amazon-reviews-input-output-1.3b" \
