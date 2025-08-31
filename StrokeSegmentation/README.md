# StrokeSegmentation

This is the project for stroke segmentation.

# Setup Environment

1. Install uv (if you didn't install uv yet)

```bash
pip install uv
```

2. Install torch for your cuda version(**skip to step 3 if you don't need to train model**)

- check your cuda version
```bash
nvcc -v
```
- install torch (use cuda 12.8 as example)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

- check whether torch is installed correctly
```bash
uv run torch_version.py
```

**ex.** How to change source for uv(换源)

- Windows:
create `/uv/uv.toml` under `%APPDATA%/`
- Linux/MacOS:
create `/uv/uv.toml` under `~/.config/`

add following lines to `uv.toml`:

```
# 使用阿里云镜像源
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
uv pip install [包名]

# 或清华大学镜像源
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
```

3. Install other dependencies

```bash
uv sync
```

4. Run Train or Predict

- run train
```bash
uv run train.py
```

- run predict
```bash
uv run predict.py
```
