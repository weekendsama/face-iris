# Multimodal Cancelable Biometrics

TensorFlow prototype for face-iris multimodal cancelable biometric template protection.

## Structure

- `src/config.py`: model and runtime config
- `src/model.py`: feature extraction, hashing, fusion model
- `src/protection.py`: template protection and matching utilities
- `src/demo.py`: runnable demo with random tensors

## Run

Use the `d2l` environment:

```powershell
& 'D:\Miniconda3\condabin\conda.bat' run -p 'D:\Miniconda\envs\d2l' python -m src.demo
```

Minimal training loop:

```powershell
& 'D:\Miniconda3\condabin\conda.bat' run -p 'D:\Miniconda\envs\d2l' python -m src.train
```

Evaluation:

```powershell
& 'D:\Miniconda3\condabin\conda.bat' run -p 'D:\Miniconda\envs\d2l' python -m src.evaluate
```

## Modes

Set `EXPERIMENT_MODE` before training or evaluation:

- `fusion`: full multimodal pipeline (default)
- `face_only`: face-only probe model and checkpoint
- `iris_only`: iris-only probe model and checkpoint

Fusion:

```powershell
$env:EXPERIMENT_MODE='fusion'
& 'D:\Miniconda3\condabin\conda.bat' run -p 'D:\Miniconda\envs\d2l' python -m src.train
& 'D:\Miniconda3\condabin\conda.bat' run -p 'D:\Miniconda\envs\d2l' python -m src.evaluate
```

Face only:

```powershell
$env:EXPERIMENT_MODE='face_only'
& 'D:\Miniconda3\condabin\conda.bat' run -p 'D:\Miniconda\envs\d2l' python -m src.train
& 'D:\Miniconda3\condabin\conda.bat' run -p 'D:\Miniconda\envs\d2l' python -m src.evaluate
```

Iris only:

```powershell
$env:EXPERIMENT_MODE='iris_only'
& 'D:\Miniconda3\condabin\conda.bat' run -p 'D:\Miniconda\envs\d2l' python -m src.train
& 'D:\Miniconda3\condabin\conda.bat' run -p 'D:\Miniconda\envs\d2l' python -m src.evaluate
```
