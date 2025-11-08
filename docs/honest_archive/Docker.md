# Docker Quickstart

CPU-only image:

```bash
make docker-build-cpu
make docker-run-cpu
```

CUDA/NVIDIA image (Linux with GPU):

```bash
make docker-build-cuda
make docker-run-cuda
```

This uses the official NVIDIA CUDA runtime base image and installs the project with the GPU extra.
Ensure that Docker is configured with GPU access (`--gpus all`).
