#!/usr/bin/sh

# TC setup
DATA_PROJECT="ads_content_understanding"
ONCALL="megataxon"
ENTITLE="ads_global_tc_ads_score"
CLUSTER="MastProdCluster"
TAG="ads_ranking_taxonomy_relevance"
MTYPE="ads_ugc_relevance"

# TC env
DISABLE_NFS=1 # disabled NFS mounting
DISABLE_OILFS=1 # disabled OILFS mounting
MANIFOLDFS_BUCKET=coin
LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so"

# running configs
model="pythia_410m" #
data_set="beauty"
set="name" # address for yelp, name for Amazon

# training specification
steps=2500
bz=2
# lr=2.25e-5
lr=1e-5
q_max_len=1024
p_max_len=64
n_epoch=30
# warmup_ratio=0.15
warmup_ratio=0.01
nnodes=8
save_steps=10
eval_steps=10

app_id='${app_id}'
model_dir="/mnt/mffuse/scaling_law/pretrained_models"
data_dir="/mnt/mffuse/scaling_law/taste_data/${model}/${data_set}"
output_dir="/mnt/mffuse/pretrain_recommendation/TC_AmazonReview/${data_set}checkpoint/${app_id}/"

resume_from_checkpoint_model="/mnt/mffuse/pretrain_recommendation/TC_AmazonReview/beautycheckpoint/conda-torchrun-li1-rncstq4//pythia_410m_beauty_1e-5_1024_64_0.01_GPUsNodes_8/name/checkpoint-1900"

report_dir="${model}_${data_set}_${lr}_${q_max_len}_${p_max_len}_${warmup_ratio}_GPUsNodes_${nnodes}"

echo "model: ${model}\n set: ${data_set}\n bz: ${bz}\n lr: ${lr}\n q_max_len: ${q_max_len}\n p_max_len: ${p_max_len}; number_of_nodes: ${nnodes}"

torchx run \
    --scheduler=mast_conda \
    --scheduler_args hpcIdentity=${DATA_PROJECT},hpcJobOncall=${ONCALL},rmAttribution=${ENTITLE},hpcClusterName=${CLUSTER},tags=${TAG},modelTypeName=${MTYPE},workspace_fbpkg_name=metaconda_demo,fbpkg_ids=manifold.manifoldfs:prod,torchx_conda_mount:stable  \
    fb.conda.torchrun \
    --env "DISABLE_NFS=${DISABLE_NFS};DISABLE_OILFS=${DISABLE_OILFS};MANIFOLDFS_BUCKET=${MANIFOLDFS_BUCKET};LD_PRELOAD=${LD_PRELOAD}" \
    --run_as_root True \
    --h zionex_80g \
    -- \
    --no-python --nnodes=${nnodes} --nproc-per-node=8 \
    ./run.sh train.py \
    --output_dir ${output_dir}/${report_dir}/${set}  \
    --resume_from_checkpoint ${resume_from_checkpoint_model}  \
    --do_train  \
    --save_steps $save_steps  \
    --eval_steps $eval_steps  \
    --train_path ${data_dir}/train_${set}.jsonl  \
    --eval_path ${data_dir}/valid_${set}.jsonl  \
    --per_device_train_batch_size $bz \
    --per_device_eval_batch_size $bz \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate $lr  \
    --q_max_len $q_max_len  \
    --p_max_len $p_max_len  \
    --warmup_ratio $warmup_ratio \
    --seed 2022  \
    --num_train_epochs $n_epoch  \
    --evaluation_strategy steps  \
    --logging_dir ${output_dir}/${report_dir}/${set}-log \
    --model_name_or_path ${model_dir}/${model}  \
