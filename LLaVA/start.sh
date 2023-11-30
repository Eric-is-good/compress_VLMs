# origin
sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list
sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
apt-get update
apt-get install -y python3-pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# pip pkg with torch
pip install /opt/data/private/eric/pkg/torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl
pip install /opt/data/private/eric/pkg/triton-2.0.0-1-cp38-cp38-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
pip install torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
apt-get install -y libopenblas-base libopenmpi-dev
# pip with others
pip install accelerate
pip install /opt/data/private/eric/pkg/bitsandbytes-0.41.0-py3-none-any.whl
pip install einops-exts==0.0.4
pip install einops==0.6.1
pip install gradio==3.35.2
pip install gradio_client==0.2.9
pip install httpx==0.24.0
pip install markdown2==2.4.10
pip install peft==0.4.0
pip install scikit-learn==1.2.2
pip install sentencepiece==0.1.99
pip install shortuuid==1.0.11
pip install timm==0.6.13
pip install transformers==4.31.0
pip install tokenizers==0.13.3
pip install wandb==0.15.12
pip install wavedrom==2.0.3.post3
pip install Pygments==2.16.1
apt purge -y libhwloc-dev libhwloc-plugins
pip install deepspeed
python3 -m pip install -U pydantic spacy
pip install -U requests
apt-get install -y screen
# 
# flash_attn
pip install /opt/data/private/eric/pkg/flash_attn-2.3.3+cu117torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
# pip install /opt/data/private/eric/pkg/flash_attn-2.3.3+cu118torch2.0cxx11abiTRUE-cp38-cp38-linux_x86_64.whl