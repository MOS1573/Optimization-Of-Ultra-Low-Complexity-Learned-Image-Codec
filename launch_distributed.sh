#!/bin/bash -l
#SBATCH --job-name=exp_name
#SBATCH --nodes=1

# For a100
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:8
#SBATCH --time=24:00:00

# Default values
EXPERIMENT="exp_name"
NUM_GPUS=8
MIXED_PRECISION="bf16"
GRADIENT_ACCUMULATION_STEPS=1
MAIN_PATH="${MAIN_PATH:-path/to/workspace}"
REPO_DIR=$MAIN_PATH"/repos/loc-lic"
CONFIG=$REPO_DIR"/configs/models/$EXPERIMENT"
BATCH_SIZE=40
EPOCHS=200
LEARNING_RATE=1e-4
DATASET="${DATASET:-path/to/dataset/images}"
DATASET_CSV=$REPO_DIR"/datasets/csv/train_256" #train_256_one_epoch
CLIP_MAX_NORM=10


cd $REPO_DIR
# Activate the conda environment
source ~/miniconda/bin/activate loc-lic


# Print configuration
echo "=================================="
echo "Distributed Training Configuration"
echo "=================================="
echo "Number of GPUs:              $NUM_GPUS"
echo "Mixed Precision:             $MIXED_PRECISION"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Config:                      $CONFIG"
echo "Experiment:                  $EXPERIMENT"
echo "Batch Size (per GPU):        $BATCH_SIZE"
echo "Test Batch Size:             $BATCH_SIZE"
echo "Effective Batch Size:        $((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Epochs:                      $EPOCHS"
echo "Learning Rate:               $LEARNING_RATE"
echo "Clip Max Norm:               $CLIP_MAX_NORM"
echo "Main Path:                   $MAIN_PATH"
echo "Dataset:                     ${DATASET:-not specified}"
echo "Dataset CSV:                 ${DATASET_CSV:-not specified}"
echo "=================================="
echo ""

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate is not installed"
    echo "Please install it: pip install accelerate"
    exit 1
fi

# Launch training
echo "Launching distributed training..."
echo ""


# Accelerate modes: eager, aot_eager, inductor, aot_ts_nvfuser, nvprims_nvfuser, cudagraphs, ofi, fx2trt, onnxrt, tensorrt, aot_torchxla_trace_once, torhchxla_trace_once, ipex, tvm

# Build the command with optional parameters
TRAIN_CMD="accelerate launch \
  --num_processes=$NUM_GPUS \
  --num_machines=1 \
  --dynamo_backend="no" \
  --mixed_precision=$MIXED_PRECISION \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  scripts/train.py \
  --config $CONFIG \
  --exp $EXPERIMENT \
  --batch_size $BATCH_SIZE \
  --test_batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --main_path $MAIN_PATH \
  --clip_max_norm $CLIP_MAX_NORM \
  --dataset $DATASET \
  --dataset_csv $DATASET_CSV \
  --mixed_precision $MIXED_PRECISION \
  --save True"

# Execute the command
eval $TRAIN_CMD

echo ""
echo "Training completed!"
