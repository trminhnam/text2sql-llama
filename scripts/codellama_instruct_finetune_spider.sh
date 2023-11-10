export BASE_DIR=.
export PROJECT_NAME=text2sql
export SESSION_NAME=codellama_finetune_spider
export PROJECT_DIR=$BASE_DIR/$PROJECT_NAME
export SESSION_DIR=$PROJECT_DIR/$SESSION_NAME

# Wandb project management
export WANDB_API_KEY=<WANDB_TOKEN>
export WANDB_PROJECT=$PROJECT_NAME
export WANDB_RUN=$SESSION_NAME

wandb login $WANDB_API_KEY

huggingface-cli login --token <HUGGINGFACE_TOKEN> --add-to-git-credential

# Description: Script for training and evaluating language adapter
export PYTHONWARNINGS="ignore"


# Model parameters
export model_name_or_path=codellama/CodeLlama-7b-hf
export peft_name_or_path=tmnam20/codellama_instructtext2sql

# peft config
export lora_r=8
export lora_target_modules=q_proj,v_proj
export lora_alpha=8
export lora_dropout=0.1
export lora_bias=lora_only

# dataset parameters
export dataset_name=tmnam20/SpiderInstruct
export preprocessing_num_workers=8
export dataloader_num_workers=8

# Training parameters
export train_batch_size=8
export learning_rate=5e-5
export num_train_epochs=5
export max_steps=-1
export max_train_samples=10000000
export gradient_accumulation_steps=32
export optim=adamw_bnb_8bit
export warmup_ratio=0.06

# Evaluation parameters
export evaluation_strategy=steps
export eval_steps=200
export eval_batch_size=32
export max_eval_samples=1000000

# logging with wandb and push to hub
export hub_model_id=$SESSION_NAME
export hub_token=<HUGGINGFACE_TOKEN>
export hub_strategy=all_checkpoints
export report_to=wandb
export logging_steps=100

# Directories
export cache_dir=$BASE_DIR/cache
export output_dir=$SESSION_DIR

# saving checkpoints
export save_steps=$eval_steps

cd ..
python finetune.py \
    --model_name_or_path $model_name_or_path \
    --dataset_name $dataset_name \
    --preprocessing_num_workers $preprocessing_num_workers \
    --dataloader_num_workers $dataloader_num_workers \
    --do_train \
    --per_device_train_batch_size $train_batch_size \
    --per_device_eval_batch_size $eval_batch_size \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --max_steps $max_steps \
    --max_train_samples $max_train_samples \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --push_to_hub \
    --hub_private_repo \
    --hub_model_id $hub_model_id \
    --hub_token $hub_token \
    --hub_strategy $hub_strategy \
    --optim $optim \
    --warmup_ratio $warmup_ratio \
    --do_eval \
    --evaluation_strategy $evaluation_strategy \
    --eval_steps $eval_steps \
    --max_eval_samples $max_eval_samples \
    --cache_dir $cache_dir \
    --report_to $report_to \
    --run_name $WANDB_RUN \
    --logging_steps $logging_steps \
    --logging_first_step \
    --hub_strategy $hub_strategy \
    --save_steps $save_steps \
    --load_best_model_at_end \
    --peft_name_or_path $peft_name_or_path \
    --load_in_4bit \
    --bnb_4bit_quant_type "nf4"\
    --bnb_4bit_compute_dtype "bf16" \
    --bnb_4bit_use_double_quant \
    --bf16 \
    --bf16_full_eval \
