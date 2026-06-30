#!/bin/bash
#SBATCH --job-name=gpt-oss-offline
#SBATCH --partition=1n4gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --time=4:00:00



# Initialize and activate conda
[ -e .venv ] || conda create -p .venv -y python=3.13 pip rust
eval "$(conda shell.bash hook)"
conda activate ./.venv

LOCKFILE=ai4s.setup.done
if ! [ -e $LOCKFILE ]; then
	pip install --upgrade pip
	./setup_enroot.sh
	CC=gcc CXX=g++ ./setup.sh
	CC=gcc CXX=g++ pip install sglang
	touch $LOCKFILE
fi

WORKDIR=$SLURM_SUBMIT_DIR
cd $WORKDIR

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TIME_STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs outputs

echo "=========================================="
echo "GPT-OSS-120B MLPerf Offline Benchmark"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi -L | wc -l) GPU(s)"
echo "=========================================="

# Start SGLang server
nohup python3 -m sglang.launch_server \
    --model-path /work/hps0/home/ea0020/other-code/inference/language/gpt-oss-120b/download/gpt-oss-model/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 30000 \
    --tensor-parallel-size 4 \
    --max-running-requests 512 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 16384 \
    --enable-metrics \
    --stream-interval 500 \
    > "logs/server_${TIME_STAMP}.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for server to be ready..."
for i in $(seq 1 300); do
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:30000/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Server ready after ${i}s (HTTP $HTTP_CODE)"
        break
    fi
    sleep 2
done

# Wait for full warmup (FlashInfer autotune, etc.)
echo "Waiting 60s for server warmup..."
sleep 60

# Run workers
python3 run_mlperf.py \
    --scenario offline \
    --input-file /work/hps0/home/ea0020/other-code/inference/language/gpt-oss-120b/download/gpt-oss-dataset/acc/acc_eval_ref.parquet \
    --backend sglang \
    --server-url http://localhost:30000 \
    --output-dir "outputs/results_${TIME_STAMP}" \
    --max-new-tokens 32768 \
    --mlperf-conf mlperf.conf \
    --user-conf user.conf || true

# Kill server and its worker processes immediately after benchmark
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Shutting down server (PID $SERVER_PID)..."
    pkill -P $SERVER_PID 2>/dev/null || true
    kill $SERVER_PID 2>/dev/null || true
    sleep 3
    kill -9 $SERVER_PID 2>/dev/null || true
fi
echo "Benchmark complete"
