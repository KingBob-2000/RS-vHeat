export nnodes=1 nrank=0 nprocs=16 mport=29501 maddr="127.0.0.1"
export pycmds="main.py --cfg configs/vHeat/vHeat_small_224.yaml --batch-size 128 --data-path /path/to/dataset --output output/vHeat_small"
python -m torch.distributed.launch --nnodes ${nnodes} --node_rank ${nrank} --nproc_per_node ${nprocs} --master_addr ${maddr} --master_port ${mport} --log_dir ${log_url} ${pycmds}