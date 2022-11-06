python3 run_clm_io.py \
    --model_name_or_path facebook/opt-6.7b \
    --dataset_name AlekseyKorshuk/amazon-reviews-input-output \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 8 \
    --output_dir /tmp/test-clm \
    --num_train_epochs 5 \
    --with_tracking

python3 run_clm_trainer.py \
    --model_name_or_path facebook/opt-350m \
    --dataset_name AlekseyKorshuk/amazon-reviews-input-output \
    --do_train \
    --do_eval \
    --fp16 \
    --logging_strategy steps \
    --logging_steps 500 \
    --logging_first_step \
    --report_to all \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 8 \
    --output_dir /tmp/test-clm \
    --num_train_epochs 5 \
    --with_tracking

