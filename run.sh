#!/bin/sh

basedir="./"

get_timestamp() {
    date +"%Y-%m-%d_%H-%M-%S"
}

save_folder="${basedir}/results/$(get_timestamp)"

model_type="ContraAttnUNET"
cuda_id=-1
batch_size=2

mkdir -p "$save_folder"

python3 validation.py \
    -save_path $save_folder \
    -model_type $model_type \
    -cuda_id $cuda_id \
    -batch_size $batch_size \
    -description "Attention-augmented UNETR" \
    -covariates \
    -rnc \
    > "${save_folder}/train_${model_type}.log" 2>&1
    # -checkpoint_path $checkpoint_path \
    # -resume_training \
    # -smoothing \
    # -template_space \