export BASE_DIR=.
export PROJECT_NAME=text2sql
export SESSION_NAME=codellama_instruct_spider
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
# export model_name_or_path=meta-llama/Llama-2-7b-hf
export peft_name_or_path=tmnam20/codellama_instruct_spider
export model_name_or_path=codellama/CodeLlama-7b-Instruct-hf

# dataset parameters
export dataset_dir=dataset/spider


# Directories
export cache_dir=$BASE_DIR/cache
export output_dir=$SESSION_DIR/output
export output_path=$output_dir/dev_predict.txt

cd ..
python predict.py \
    --model_name_or_path $model_name_or_path \
    --peft_name_or_path $peft_name_or_path \
    --dataset_dir $dataset_dir \
    --output_dir $output_dir \
    --output_path $output_path \
    --cache_dir $cache_dir \
    --load_in_4bit \
    --bnb_4bit_quant_type "nf4"\
    --bnb_4bit_compute_dtype "bf16" \
    --bnb_4bit_use_double_quant \
    --bf16 \
    --bf16_full_eval \
    --use_llama_prompt \
