#!/bin/bash
# Train script.
set -eux

name=$1
# gpus=$2
NODE_RANK=$2

export MASTER_PORT=23857
# export WORK_DIR=`pwd`
ROOT_DIR=~/work/transpeeder
WORK_DIR=${ROOT_DIR}/examples

#ds_report

OUTPUT=${WORK_DIR}/output/${name}
if [ -d $OUTPUT ]; then
    # rm
    echo "${OUTPUT} exist."
else
    mkdir -p ${OUTPUT}
fi

NODE_0_IP=192.168.30.214
NODE_1_IP=192.168.30.21
# NODE_0_NUM_DEVICES=2
# NODE_1_NUM_DEVICES=2

export MASTER_ADDR=${NODE_0_IP}


HOST_FILE=${WORK_DIR}/host.txt

# deepspeed --host ${NODE_0_IP}:${NODE_0_NUM_DEVICES},${NODE_1_IP}:${NODE_1_NUM_DEVICES} \
deepspeed --hostfile=${HOST_FILE} --no_ssh \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} ${WORK_DIR}/train.py \
    --output_dir ${OUTPUT} \
    --init_ckpt /home/jongwonlee/work/transpeeder/ckpt/llama-3.2-1B-mp-2/ \
    --data_path /home/jongwonlee/work/transpeeder/data/alpaca_data_sample_oneline_format.json \
    --max_seq_len 8192 \
    --train_steps 1000 \
    --eval_steps 10 \
    --save_steps 200 \
    --log_steps 1 \
    --pipe_parallel_size 2 \
    --model_parallel_size 2 \
    --use_flash_attn false \
    --ntk false \
    --deepspeed_config ${ROOT_DIR}/configs/ds_config_llama_32_1b.json