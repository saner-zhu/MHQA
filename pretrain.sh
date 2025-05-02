nproc_per_node=4
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift pt \
    --model bart-base \
    --train_type full \
    --dataset swift/er_extract \
    --torch_dtype bfloat16 \
    --streaming true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --packing true \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed zero3 \
    --max_length 2048 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --output_dir output/bart-base \
    --attn_impl flash_attn