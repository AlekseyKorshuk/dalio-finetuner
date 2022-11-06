python3 run_clm_io.py \
    --model_name_or_path facebook/opt-350m \
    --dataset_name AlekseyKorshuk/amazon-reviews-input-output \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --output_dir /tmp/test-clm \
    --num_train_epochs 1 \
    --with_tracking \
    --overwrite_cache
