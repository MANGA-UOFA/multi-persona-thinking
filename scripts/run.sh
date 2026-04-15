SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_DIR=$(dirname "$SCRIPT_DIR")

echo "Targeting Root Directory: $ROOT_DIR"
cd "$ROOT_DIR"

model="meta-llama/Llama-3.1-8B-Instruct"
method=mpt
dataset=BBQ

echo ">>> Running  method = $method"
python mpt_inference.py --method $method --models $model  --dataset $dataset --chat_templates 
python mpt_evaluate.py --method $method --models $model --dataset $dataset --chat_templates