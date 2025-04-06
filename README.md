# CUDA PRACTICE

> cuda best practice &amp; notes

this is a general template for customizing some CUDA operators to integrate into your own training framwork to achieve acceleration effects.

choose the right cuda/torch version, the gpu used in this project is A10, so the highest version of nvidia driver is 12.2, you can choose the version that suits your device.

```
micromamba install -c conda-forge cuda-toolkit=12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## pybind version

the first way to bind cuda kernels to python is using `pybind11`.
you can use `pybind11` to bind cuda kernels to python, and then import them in python to use.

Run `pip install -e .` to use the kernels in the `prtc` namespace. You can then import `prtc` from the tests directory to execute and validate these kernels.

When commit and push code, pleace use `pre-commit` to ensure consistent code formatting style

```
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
s
modify `setup.py` to include 3rd libs, such as `ThunderKittens` or `cutlass`, but remember to match your CUDA version, otherwise there will be dependency issues, for example, the minimum cuda version for `ThunderKittens` is 12.3 or higher...

## jit version

the second and recommended way to bind cuda kernels to python is using jit. it is also suitable for c++ projects.
