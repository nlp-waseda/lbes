# Likelihood-Based Evolution Strategies（LBES）

論文公開用の実装です。`vLLM + Ray` 上で、モデルパラメータへのノイズを用いた **Evolution Strategies (ES)** と、その派生である **Likelihood-Based Evolution Strategies (LBES)** を実行します。

- ES: ノイズを加えた複数個体で生成 -> 報酬 -> 更新
- LBES: 生成したサンプルの「報酬」と「尤度」の相関に基づいて ES 更新

## セットアップ

### 要件

- Linux
- Python >= 3.12
- CUDA 対応 GPU

### インストール（uv）

```bash
git clone https://github.com/nlp-waseda/lbes.git
cd lbes
uv sync
```

---

## データ

学習/評価データは `parquet` で、各行が以下のカラムを持ちます。

- `prompt`: Chat 形式のプロンプト（`list[{"role": str, "content": str}]`）
- `reward_func_args`: 報酬関数に渡す引数（`dict[str, Any]`）

`data/` 配下に、サンプル（GSM8K と カウントダウンタスク）の `train.parquet` / `test.parquet` が同梱されています。

### 付属の前処理スクリプト

- GSM8K: `src/data_preprocessing/gsm8k.py`

```bash
uv run src/data_preprocessing/gsm8k.py
```

- カウントダウンタスク: `src/data_preprocessing/countdown.py`

```bash
uv run src/data_preprocessing/countdown.py --n-given-numbers 3
```

---

## 報酬関数

報酬関数はドット区切りパスで指定し、以下の形の関数を実装します。

```python
def compute_score(completion_text: str, **kwargs) -> dict[str, float]:
  ...
```

重要:

- 戻り値の dict には必ず `"reward"` キーを含めてください（学習に使用）。
- `**kwargs` には `reward_func_args` の中身がそのまま渡されます。

例:

- GSM8K: `src/reward_funcs/gsm8k.py`
- カウントダウンタスク: `src/reward_funcs/countdown.py`

---

## 実行方法

エントリポイントは以下です。

- ES: `src/es.py`（設定: `src/es_config.py`）
- LBES: `src/lbes.py`（設定: `src/lbes_config.py`）

どちらも `transformers.HfArgumentParser` 経由で dataclass を CLI 引数にマップしています。

### 最小実行例（ローカル）

#### ES（GSM8K）

```bash
uv run src/es.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir runs/es-gsm8k \
    --train-file data/gsm8k/train.parquet \
    --eval-file data/gsm8k/test.parquet \
    --reward-func-path src.reward_funcs.gsm8k.compute_score \
    --tensor-parallel-size 1 \
    --population-size 32 \
    --data-batch-size 64 \
    --n 1 \
    --temperature 0.0 \
    --max-tokens 1024 \
    --noise-std 1e-3 \
    --learning-rate 5e-7 \
    --num-train-epochs 1 \
    --eval-strategy steps \
    --eval-steps 50 \
    --save-strategy steps \
    --save-steps 200
```

#### LBES（GSM8K）

```bash
uv run src/lbes.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir runs/lbes-gsm8k \
    --train-file data/gsm8k/train.parquet \
    --eval-file data/gsm8k/test.parquet \
    --reward-func-path src.reward_funcs.gsm8k.compute_score \
    --tensor-parallel-size 1 \
    --population-size 32 \
    --data-batch-size 64 \
    --n 8 \
    --temperature 0.1 \
    --max-tokens 1024 \
    --noise-std 1e-3 \
    --learning-rate 5e-7 \
    --flat-corrcoef True \
    --top-k-prompt-reward-var 32 \
    --top-k-rewards 1 \
    --bottom-k-rewards 1 \
    --num-train-epochs 1 \
    --eval-strategy steps \
    --eval-steps 50 \
    --save-strategy steps \
    --save-steps 200
```

## [TSUBAME 4.0](https://www.t4.cii.isct.ac.jp/)におけるジョブスクリプト例

### ES（GSM8K）

```bash
#!/bin/bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=5:00:00
#$ -j y

module purge
module load cuda/12.8.0 cudnn/9.8.0 nccl/2.26.2

# .envにてHF_TOKEN、WANDB_API_KEYを設定している想定
set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<HF_HOME>
export WANDB_PROJECT=es-gsm8k
export run_name=<RUN_NAME>
export model=meta-llama/Llama-3.1-8B-Instruct
export output_dir=<OUTPUT_DIR>
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PATH="$HOME/.local/bin:$PATH"

uv run src/es.py \
    --model $model \
    --max-model-len 1418 \
    --n 1 \
    --temperature 0.0 \
    --max-tokens 1024 \
    --output-dir $output_dir \
    --train-file data/gsm8k/train.parquet \
    --eval-file data/gsm8k/test.parquet \
    --reward-func-path src.reward_funcs.gsm8k.compute_score \
    --data-batch-size 64 \
    --population-size 32 \
    --noise-std 1e-3 \
    --learning-rate 5e-7 \
    --num-train-epochs 1 \
    --logging-steps 1 \
    --eval_strategy steps \
    --eval_steps 1 \
    --report-to wandb \
    --run-name $run_name
```

### LBES（GSM8K）

```bash
#!/bin/bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=2:00:00
#$ -j y

module purge
module load cuda/12.8.0 cudnn/9.8.0 nccl/2.26.2

# .envにてHF_TOKEN、WANDB_API_KEYを設定している想定
set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<HF_HOME>
export WANDB_PROJECT=lbes-gsm8k
export run_name=<RUN_NAME>
export model=meta-llama/Llama-3.1-8B-Instruct
export output_dir=<OUTPUT_DIR>
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PATH="$HOME/.local/bin:$PATH"

uv run src/lbes.py \
    --model $model \
    --max-model-len 1419 \
    --n 8 \
    --temperature 0.1 \
    --max-tokens 1024 \
    --output-dir $output_dir \
    --train-file data/gsm8k/train.parquet \
    --eval-file data/gsm8k/test.parquet \
    --reward-func-path src.reward_funcs.gsm8k.compute_score \
    --data-batch-size 64 \
    --population-size 32 \
    --noise-std 1e-3 \
    --learning-rate 5e-7 \
    --top-k-prompt-reward-var 32 \
    --top-k-rewards 1 \
    --bottom-k-rewards 1 \
    --flat-corrcoef True \
    --num-train-epochs 1 \
    --logging-steps 1 \
    --eval-strategy steps \
    --eval-steps 1 \
    --report-to wandb \
    --run-name $run_name
```

### ES（カウントダウンタスク）

```bash
#!/bin/bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00
#$ -j y

module purge
module load cuda/12.8.0 cudnn/9.8.0 nccl/2.26.2

# .envにてHF_TOKEN、WANDB_API_KEYを設定している想定
set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<HF_HOME>
export WANDB_PROJECT=es-countdown3
export run_name=<RUN_NAME>
export model=meta-llama/Llama-3.1-8B-Instruct
export output_dir=<OUTPUT_DIR>
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PATH="$HOME/.local/bin:$PATH"

uv run src/es.py \
    --model $model \
    --max-model-len 2242 \
    --n 1 \
    --temperature 0.0 \
    --max-tokens 2048 \
    --output-dir $output_dir \
    --train-file data/countdown3/train.parquet \
    --eval-file data/countdown3/test.parquet \
    --reward-func-path src.reward_funcs.countdown.compute_score \
    --data-batch-size 64 \
    --population-size 32 \
    --noise-std 1e-3 \
    --learning-rate 5e-7 \
    --num-train-epochs 2 \
    --logging-steps 1 \
    --eval_strategy steps \
    --eval_steps 1 \
    --report-to wandb \
    --run-name $run_name
```

### LBES（カウントダウンタスク）

```bash
#!/bin/bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=10:00:00
#$ -j y

module purge
module load cuda/12.8.0 cudnn/9.8.0 nccl/2.26.2

# .envにてHF_TOKEN、WANDB_API_KEYを設定している想定
set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<HF_HOME>
export WANDB_PROJECT=lbes-countdown3
export run_name=<RUN_NAME>
export model=meta-llama/Llama-3.1-8B-Instruct
export output_dir=<OUTPUT_DIR>
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PATH="$HOME/.local/bin:$PATH"

uv run src/lbes.py \
    --model $model \
    --max-model-len 2243 \
    --n 8 \
    --temperature 0.1 \
    --max-tokens 2048 \
    --output-dir $output_dir \
    --train-file data/countdown3/train.parquet \
    --eval-file data/countdown3/test.parquet \
    --reward-func-path src.reward_funcs.countdown.compute_score \
    --data-batch-size 64 \
    --population-size 32 \
    --noise-std 1e-3 \
    --learning-rate 5e-7 \
    --top-k-prompt-reward-var 32 \
    --top-k-rewards 1 \
    --bottom-k-rewards 1 \
    --flat-corrcoef True \
    --num-train-epochs 2 \
    --logging-steps 1 \
    --eval-strategy steps \
    --eval-steps 1 \
    --report-to wandb \
    --run-name $run_name
```

---

## 主要な CLI 引数（抜粋）

### 生成・実行（vLLM EngineArgs / SamplingParams）

- `--model`: Hugging Face のモデル ID / ローカルパス
- `--load-format`: vLLM のロード形式（デフォルト: `auto`）
- `--max-model-len`: 最大コンテキスト長
- `--tensor-parallel-size`: テンソル並列の並列度
- `--gpu-memory-utilization`: vLLM の GPU メモリ利用率（デフォルト: 0.9）
- `--n`: 1 プロンプトあたりのサンプル数
- `--temperature`, `--max-tokens`

### 学習（共通: ES/LBES）

- `--train-file`, `--eval-file`: parquet
- `--reward-func-path`: 例 `src.reward_funcs.gsm8k.compute_score`
- `--data-batch-size`: データローダのバッチサイズ
- `--population-size`: 個体数
- `--noise-std`: ノイズの標準偏差
- `--learning-rate`: 学習率
- `--num-train-epochs`

### LBES 固有: 報酬に基づくフィルタ

LBES は各ステップで、(1) まず `n` 個の生成と報酬計算を行い、(2) 報酬に基づいてバッチ内のプロンプト/生成を **フィルタ** してから、(3) フィルタ後のサンプルに対してノイズ下の平均対数尤度を計算し、相関で更新量を決めます。

フィルタの結果、すべてのサンプルが除外されると、そのステップはスキップされます。

- `--top-k-prompt-reward-var`:
  - バッチ内の各プロンプトについて、`n` 個の報酬の分散を計算し、分散が大きい上位 `k` プロンプトのみ残します（`k` は `data_batch_size` を上限に切り詰め）。
  - 同点は乱数でタイブレークします。
- `--top-k-rewards`:
  - 各プロンプト内で報酬上位 `k` 個の生成のみ残します。
- `--bottom-k-rewards`:
  - 各プロンプト内で報酬下位 `k` 個の生成のみ残します。
  - `top-k` と `bottom-k` は併用可能で、集合としてマージされます（重複は除去）。
- `--exclude-same-rewards-samples`:
  - 上記で選ばれた生成の報酬がすべて同一（`np.allclose`）になったプロンプトは丸ごと捨てます。
- `--flat-top-k-rewards`, `--flat-bottom-k-rewards`:
  - （上のフィルタ後に）バッチ内の全報酬をフラットに並べ、全体で上位/下位 `k` 個だけを残します。
  - その結果、あるプロンプトが 1 つも残らない場合は、そのプロンプト自体が除外されます。
  - 同点は乱数でタイブレークします。

補足:

- `--flat-corrcoef` はフィルタではなく「相関の取り方」を切り替えるフラグです。
  - `False`: 各プロンプト内で中心化した報酬と対数尤度を用いて相関を計算し、プロンプト平均を取ります（この場合 `n>=2` が必要）。
  - `True`:報酬と対数尤度をプロンプト次元も含めてフラット化して相関を計算します。

### スケジューラ

`noise_std` と `learning_rate` はスケジューラを指定できます。

- `--noise-std-scheduler-type`: `constant|linear|cosine|exponential|polynomial`
- `--noise-std-final`: linear/cosine/polynomial の終端値
- `--lr-scheduler-type`: `constant|linear|cosine|exponential|polynomial`
- `--lr-final`: linear/cosine/polynomial の終端値

### 評価・保存

- `--eval-strategy`: `no|steps|epoch`
- `--eval-steps`: `steps` の場合に必須
- `--save-strategy`: `no|steps|epoch`
- `--save-steps`: `steps` の場合に必須
- `--save-total-limit`: チェックポイント保持数（古いものから削除）

### ログ（Weights & Biases）

- `--report-to wandb` で `wandb.log` が有効になります
- `--run-name` で実験名を指定できます

---

## 出力

`--output-dir` に以下が保存されます。

- 最終モデル（`model.safetensors` など。TP>1 ではシャード保存）
- `--save-strategy` 有効時: `checkpoint-<step>/`

※ TP>1 の場合、モデルはシャード（分割）して保存されます。

---

## 自分のタスクを追加する

1. `parquet` を作る（各行に `prompt` と `reward_func_args`）
2. `compute_score(completion_text, **reward_func_args) -> dict[str, float]` を実装し、`reward` を必ず返す
3. `--train-file/--eval-file/--reward-func-path` を指定して `src/es.py` か `src/lbes.py` を実行

<!-- ---

## 引用（Citation）

```bibtex

```

## ライセンス

`LICENSE` を参照してください。
-->
