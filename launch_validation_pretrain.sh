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


nnodes=1


app_id='${app_id}'
data_dir="/mnt/mffuse/scaling_law/taste_data/${model}/"

experiment_name="/mnt/mffuse/pretrain_recommendation/Proposed/fineweb_contrastive_checkpoint/conda-torchrun-li1-m0m3p6d/pythia_410m_fineweb_contrastive_1e-5_2048_64_0.01/"
echo "right now running validation on ${experiment_name}"

torchx run \
    --scheduler=mast_conda \
    --scheduler_args hpcIdentity=${DATA_PROJECT},hpcJobOncall=${ONCALL},rmAttribution=${ENTITLE},hpcClusterName=${CLUSTER},tags=${TAG},modelTypeName=${MTYPE},workspace_fbpkg_name=metaconda_demo,fbpkg_ids=manifold.manifoldfs:prod,torchx_conda_mount:stable  \
    fb.conda.torchrun \
    --env "DISABLE_NFS=${DISABLE_NFS};DISABLE_OILFS=${DISABLE_OILFS};MANIFOLDFS_BUCKET=${MANIFOLDFS_BUCKET};LD_PRELOAD=${LD_PRELOAD}" \
    --run_as_root True \
    --h zionex_80g \
    -- \
    --no-python --nnodes=${nnodes} --nproc-per-node=8 \
    ./run.sh evaluate_torchx.py  \
    --data_dir ${data_dir} \
    --data_name beauty  \
    --experiment_name ${experiment_name} \
    --seed 2022  \
    --item_size 64  \
    --seq_size 1024  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 4  \
    --stopping_step 5  \
    --all_models_path ${experiment_name}/name
