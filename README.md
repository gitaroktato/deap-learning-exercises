## CUDA and cuDNN install for WSL
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
https://developer.nvidia.com/cudnn-downloads

## CUDA on WSL guides
https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2

```shell
docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit


Testing - Docker
https://docs.docker.com/desktop/gpu/
```shell
docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

# A simple way to check CUDA and cuDNN - Python
https://stackoverflow.com/questions/76229252/cudnn-error-when-running-jax-on-gpu-with-apptainer

```python
import jax
jax.numpy.array(1.)
```

# Docker on WSL2
https://nickjanetakis.com/blog/install-docker-in-wsl-2-without-docker-desktop
https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute

```shell
curl -fsSL https://get.docker.com | bash
dockerd-rootless-setuptool.sh install
sudo apt-get install -y uidmap
```

Run docker rootless
```shell
PATH=/usr/bin:/sbin:/usr/sbin:$PATH dockerd-rootless.sh &
```

# cuDNN 
Check if the library is on the machine
```shell
sudo find / -name 'libcudnn.so.*'
```

In some cases you need to set the `LD_LIBRARY_PATH`
https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html
```shell
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcudnn.so
```

Testing with `nvcc`
```shell
sudo apt install nvidia-cuda-toolkit
```
```shell
nvcc -V
```