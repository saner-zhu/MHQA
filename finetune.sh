CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model DeepSeek-R1-Distill-Qwen-1.5B \
    --train_type lora \
    --dataset dataset_finetune/drr_train_dicider.jsonl \
    --torch_dtype float16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 3072 \
    --output_dir output/dicider \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --model_author swift \
    --model_name swift-root \

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model DeepSeek-R1-Distill-Qwen-1.5B \
    --train_type lora \
    --dataset dataset_finetune/drr_train_rewriter.jsonl \
    --torch_dtype float16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 3072 \
    --output_dir output/rewriter \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --model_author swift \
    --model_name swift-root \

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model DeepSeek-R1-Distill-Qwen-1.5B \
    --train_type lora \
    --dataset dataset_finetune/drr_train_responser.jsonl \
    --torch_dtype float16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 3072 \
    --output_dir output/responser \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --model_author swift \
    --model_name swift-root \
