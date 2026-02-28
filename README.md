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
& 'D:\miniconda\condabin\conda.bat' run -n d2l python -m src.demo
```

Minimal training loop:

```powershell
& 'D:\miniconda\condabin\conda.bat' run -n d2l python -m src.train
```
