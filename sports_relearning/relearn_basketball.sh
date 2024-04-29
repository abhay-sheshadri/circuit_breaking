python finetune.py \
    --model_name_or_path="PhillipGuo/gemma-2b_Unlearning_basketball" \
    --tokenizer_name="google/gemma-2b" \
    --train_file="basketball_retrain_dataset.json" \
    --output_dir="basketball_gemma_relearned" --do_train \
    --per_device_train_batch_size=8 --num_train_epochs=1 \
    --use_lora=True --lora_rank=16 \
    --torch_dtype=bfloat16 --bf16=True \
    --overwrite_output_dir \
    --log_level="warning" --logging_steps=1