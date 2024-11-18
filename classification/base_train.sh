#export nnodes=2 nrank=0 nprocs=16 mport=29501 maddr="127.0.0.1"
#export pycmds="main.py --cfg configs/vHeat/vHeat_windows_base_224.yaml --batch-size 12 --data-path /mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vmamba/classification/data_process/AID/ --output output/vHeat_base"
#CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nnodes ${nnodes} --node_rank ${nrank} --nproc_per_node ${nprocs} --master_addr ${maddr} --master_port ${mport} --log_dir ${log_url} ${pycmds}
#CONFIG=configs/vHeat/vHeat_flux_base_224.yaml
CONFIG=configs/vHeat/vHeat_flux_base_224.yaml
#DATA=/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vmamba/classification/data_process/NWPU-RESISC45/
DATA=/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/dataset_yhc/AID/
#DATA=/mnt/AIService/FoundationModel/DroneModel/dataset0702/NWPU-RESISC45/
GPUS=8
PORT=10021

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch  \
--nnodes=$NNODES     --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR  --nproc_per_node=$GPUS \
--master_port=12011 $(dirname "$0")/main.py  --cfg $CONFIG --data-path $DATA --batch-size 180 \
--output vheat_aid --tag vheat_aid   --launcher pytorch ${@:3}

 

