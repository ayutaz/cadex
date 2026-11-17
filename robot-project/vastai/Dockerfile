# vast.ai で Cadex の training/cadex_train.py（Brax PPO on MJX/JAX）を回すイメージ。
#
# ビルドは **Cadex リポジトリの根** をコンテキストにする（training/ と src/Mod/cadex/
# の一部を COPY するため）:
#
#   cd <cadex の根>
#   docker build --platform linux/amd64 -f robot-project/vastai/Dockerfile \
#       -t <registry>/cadex-train:<tag> .
#   docker push <registry>/cadex-train:<tag>
#
# --- clone ではなく COPY にした理由 ------------------------------------------
# cadex_train.py がモジュールの最上位で import するのは標準ライブラリだけで、
# 実行時に読むのは jax / mujoco / mujoco.mjx / numpy / brax / flax / optax と、
# 同じディレクトリの cadex_goto.py（--goto、地形、ジャンプコースなど）だけ。
# cadex_goto.py は地形の箱を作る時に限り `<training の親>/src/Mod/cadex/` を
# sys.path に足して CadexDynamics を読み、CadexDynamics は CadexGait と
# CadexEvaluation を関数の中で遅延 import する（3 つとも標準ライブラリ + 遅延の
# numpy/mujoco だけで、FreeCAD もビルドも要らない）。
# したがって必要なのは 5 ファイルで、FreeCAD 由来のリポジトリ全体（大きく、
# 非公開なら箱に認証情報が要る）を clone する理由が無い。代わりに、どの学習器が
# 走ったかを辿れるよう、cadex_train.py の sha256 を /opt/cadex/TRAINER_SHA256 に
# 残す（ポリシーにも trainer_sha256 が記録される。`python -m vastai check` が手元と比べる）。
# 配置は `/opt/cadex/training/` と `/opt/cadex/src/Mod/cadex/` で、Cadex の
# ディレクトリ構成どおり（cadex_goto の parents[1]/src/Mod/cadex がそのまま当たる）。
#
# ライセンス: COPY するファイルは LGPL-2.1-or-later（Cadex）。イメージを公開
# レジストリに置くなら、その条件（ソースの提供）に従うこと。非公開レジストリ推奨。
#
# vast.ai の SSH 起動（runtype ssh）ではイメージの ENTRYPOINT/CMD は使われず、
# Dockerfile の ENV も SSH のシェルには引き継がれないことがある。そのため
# スクリプト側は /opt/venv/bin/python を絶対パスで呼び、JAX_PLATFORMS などは
# 起動コマンドで明示的に渡す。
# ---------------------------------------------------------------------------

# CUDA 12 系の公式イメージ。jax[cuda12] は CUDA のライブラリを pip の nvidia-*
# パッケージで持ち込むので、base（ドライバ以外の CUDA を含まない最小構成）で足りる。
# Ubuntu 24.04 の python3 は 3.12（cadex_train の要件 >=3.12）。
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# rsync: 手元との送受信（リモート側にも rsync が要る）
# openssh-server: vast.ai の SSH 起動用
# coreutils の sha256sum / nohup、util-linux の setsid は base に含まれる
# apt の版は Ubuntu の更新で消えるので固定しない（再現性は pip のピン留めで担保）
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates rsync openssh-server procps \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version"

# 依存（ピン留め）を先に入れて、学習器の差し替えでこの層を作り直さないようにする。
# requirements.txt の jax==0.7.2 を入れたあと、同じ版の CUDA 12 版に差し替える
# （training/requirements.txt の末尾の指示どおり）。
COPY training/requirements.txt /opt/cadex/training/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install -r /opt/cadex/training/requirements.txt \
 && /opt/venv/bin/pip install "jax[cuda12]==0.7.2" \
 && /opt/venv/bin/python -c "import importlib.metadata as m; \
pins = {'mujoco': '3.10.0', 'mujoco-mjx': '3.10.0', 'jax': '0.7.2', 'numpy': '2.5.1', \
        'brax': '0.14.2', 'flax': '0.12.0', 'optax': '0.2.8'}; \
bad = {n: m.version(n) for n, v in pins.items() if m.version(n) != v}; \
assert not bad, bad"

# 学習器と、cadex_goto が地形で読むエンジンの純 Python モジュール
COPY training/cadex_train.py training/cadex_goto.py /opt/cadex/training/
COPY src/Mod/cadex/CadexDynamics.py src/Mod/cadex/CadexGait.py src/Mod/cadex/CadexEvaluation.py \
     /opt/cadex/src/Mod/cadex/
WORKDIR /opt/cadex/training
RUN sha256sum /opt/cadex/training/cadex_train.py | cut -c1-64 > /opt/cadex/TRAINER_SHA256 \
 && /opt/venv/bin/python -c "import cadex_train, cadex_goto; print('trainer', cadex_train.TRAINER_SHA256)" \
 && JAX_PLATFORMS=cpu /opt/venv/bin/python -c "import jax, mujoco, mujoco.mjx, brax, flax, optax; print('deps ok', jax.__version__, mujoco.__version__)"

# JIT コンパイルのキャッシュ（cadex_train が CADEX_JAX_CACHE を読む）。/workspace は
# vast.ai でインスタンスのディスクに載るので、停止→再開でもキャッシュが残る。
ENV CADEX_JAX_CACHE=/workspace/cache/jax
RUN mkdir -p /workspace/runs /workspace/cache/jax

WORKDIR /workspace
