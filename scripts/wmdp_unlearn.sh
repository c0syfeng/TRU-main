#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

per_device_train_batch_size=8 #4
gradient_accumulation_steps=2
learning_rate=1e-5

model=zephyr-7b-beta

data_splits=(
    "bio"
    "cyber"
)

trainers=(
    "TRU"
    "GradAscent"
    "GradDiff"
    "WGA"
    "NPO"
    "RMU"
    "PO"
)

for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do

            task_name=wmdp_${model}_${data_split}_${trainer}

            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/wmdp/default.yaml \
                model=${model} \
                data_split=${data_split} \
                trainer=${trainer} \
                task_name=${task_name} \
                paths.output_dir=saves/unlearn/${task_name}/ \
                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                trainer.args.learning_rate=${learning_rate} \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true
    done
done