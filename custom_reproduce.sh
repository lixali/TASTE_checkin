#!/bin/zsh

source ~/local/anaconda3/etc/profile.d/conda.sh
conda activate taste

set=$1

echo "---running reproduce experiment for set ${set}---"
bash reproduce/train/${set}/train_${set}.sh
echo "---${set} reproduce experiment finishes---"
