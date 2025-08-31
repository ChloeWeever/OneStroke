# StrokeSegmentation

This is the project for stroke segmentation.

# Setup Environment

1. Install uv (if you didn't install uv yet)

```bash
pip install uv
```

2. Install torch for your cuda version

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
