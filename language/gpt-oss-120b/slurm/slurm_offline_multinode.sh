#!/bin/bash
#SBATCH --job-name=gpt-oss-multinode
#SBATCH --partition=4n4gpu
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --time=8:00:00


module load nvhpc-hpcx

# (1) Environment variables to enable CUDA-aware MPI
export OMPI_MCA_pml=ucx
export UCX_CUDA_COPY_DMABUF=no
export UCX_MAX_RNDV_RAILS=4
export NCCL_DMABUF_ENABLE=0
export NCCL_NET_GDR_LEVEL=SYS

# (2) Required when using GPU-buffer communication across nodes.
#     Without this, inter-node non-blocking GPU communication
#     (MPI_Isend/Irecv, etc.) fails with
#     "cannot find remote protocol ... tag_send ... from cuda".
export UCX_PROTO_ENABLE=n

WORKDIR=$SLURM_SUBMIT_DIR
cd $WORKDIR

# Initialize and activate conda
[ -e .venv ] || conda create -p .venv -y python pip
source ~/.bashrc
conda activate ./.venv


# rm -rf .venv
exit

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=logs/nccl_debug.log

# === Communication Instrumentation ===
export HBM_LOG_FILE=logs/http_payloads.jsonl

mkdir -p logs outputs

echo "=========================================="
echo "GPT-OSS-120B MLPerf Offline (4-Node)"
echo "Node: $(hostname)"
echo "Node Rank: $SLURM_NODEID"
echo "Total Nodes: $SLURM_NNODES"
echo "GPUs per node: $(nvidia-smi -L | wc -l)"
echo "=========================================="

# Leader node detection
export LEADER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export NCCL_INIT_ADDR="${LEADER_NODE}:25000"

echo "Leader node:  $LEADER_NODE"
echo "Node rank:    $SLURM_NODEID"
echo "NCCL init:    $NCCL_INIT_ADDR"

# -------------------------------------------------
# 1. Start SGLang servers on ALL nodes
# -------------------------------------------------
echo "Starting SGLang server (TP=16, NNodes=4, NodeRank=$LEADER_NODE)..."

export TIME_STAMP=$(date +%Y%m%d_%H%M%S)

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MPIRUN_ARGS=(
	--map-by node:PE=144
	-x LEADER_NODE
	-x NCCL_INIT_ADDR
	-x OMPI_MCA_pml
	-x UCX_CUDA_COPY_DMABUF
	-x UCX_MAX_RNDV_RAILS
	-x NCCL_DMABUF_ENABLE
	-x NCCL_NET_GDR_LEVEL
	-x UCX_PROTO_ENABLE
	-x TIME_STAMP
	-x SLURM_NNODES
)

mpirun "${MPIRUN_ARGS[@]}" bash -x -c '
python3 -m sglang.launch_server \
	--model-path /work/hps0/home/ea0020/other-code/inference/language/gpt-oss-120b/download/gpt-oss-model/gpt-oss-120b \
	--host 0.0.0.0 \
	--port 30000 \
	--tensor-parallel-size 4 \
	--nnodes "${OMPI_COMM_WORLD_SIZE}" \
	--node-rank "${OMPI_COMM_WORLD_RANK}" \
	--nccl-init-addr "${NCCL_INIT_ADDR}" \
	--max-running-requests 512 \
	--mem-fraction-static 0.85 \
	--chunked-prefill-size 16384 \
	--enable-metrics \
	--stream-interval 500 \
	> "logs/server_${TIME_STAMP}_rank${OMPI_COMM_WORLD_RANK}.log" 2>&1 &
SERVER_PID=$!
echo "[Rank ${OMPI_COMM_WORLD_RANK}] Server PID: $SERVER_PID"
wait
' &

# -------------------------------------------------
# 2. Leader: wait for readiness + warmup
# -------------------------------------------------
echo "[Leader] Waiting for server to be ready..."
for i in $(seq 1 600); do
    # curl -s -w '%{http_code}' http://${LEADER_NODE}:30000/health
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://${LEADER_NODE}:30000/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[Leader] Server ready after ${i}s (HTTP $HTTP_CODE)"
        break
    fi
    sleep 2
done

echo "[Leader] Waiting 60s for warmup (FlashInfer + NCCL)..."
sleep 60

# Save warmup NCCL entries; clear for benchmark
if [ -f logs/nccl_debug.log ]; then
    cp logs/nccl_debug.log logs/nccl_debug_warmup.log
    : > logs/nccl_debug.log
fi

exit 

# Wait on all ranks until leader finishes warmup
srun --nodes=4 --ntasks-per-node=1 bash -c \
    'if [ "$SLURM_NODEID" -eq 0 ]; then sleep 90; fi; sleep 30' || true

# -------------------------------------------------
# 3. Leader: run MLPerf benchmark
# -------------------------------------------------
echo "[Leader] Running MLPerf benchmark..."

# Clear HTTP payload log from any pre-benchmark traffic
: > logs/http_payloads.jsonl

python3 run_mlperf.py \
    --scenario offline \
    --input-file /work/hps0/home/ea0020/other-code/inference/language/gpt-oss-120b/download/gpt-oss-dataset/acc/acc_eval_ref.parquet \
    --backend sglang \
    --server-url "http://${LEADER_NODE}:30000" \
    --output-dir "outputs/results_$(date +%Y%m%d_%H%M%S)" \
    --max-new-tokens 32768 \
    --mlperf-conf mlperf.conf \
    --user-conf user.conf || true

echo "[Leader] Benchmark complete"


echo "Multinode job finished"
