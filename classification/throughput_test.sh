export nnodes=1 nrank=0 nprocs=1 mport=29501 maddr="127.0.0.1"
export pycmds="main.py --cfg configs/vHeat/vHeat_flux_base_224.yaml
--batch-size 1 --throughput --data-path DATA_PATH --output OUTPUT_PATH --model_ema False  --amp-opt-level O0"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes ${nnodes} --node_rank ${nrank} --nproc_per_node ${nprocs} --master_addr ${maddr} --master_port ${mport} ${pycmds}