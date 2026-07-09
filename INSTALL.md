# Installation

This repository is built in PyTorch 2.0.1 and tested on Ubuntu 20.04 environment (Python3.9, CUDA11.8).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/zhouzhaorun/WAVSSNet.git
cd WAVSSNet
```

2. Make conda environment
```
conda create -n wavssnet python=3.9
conda activate wavssnet
```

3. Install dependencies
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```