# ComfyUI-ViViD
a comfyui custom node for [ViViD](https://github.com/alibaba-yuanjing-aigclab/ViViD),
there is a bug waiting for debug,[ViViD/issues/10](https://github.com/alibaba-yuanjing-aigclab/ViViD/issues/10)
<div>
  <figure>
  <img alt='webpage' src="web.png?raw=true" width="600px"/>
  <figure>
</div>

## how to use
make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
## insatll xformers match your torch,for torch==2.1.0+cu121
pip install xformers==0.0.22.post7
pip install accelerate 
# in ComfyUI/custom_nodes
git clone https://github.com/AIFSH/ComfyUI-ViViD.git
cd ComfyUI-ViViD
pip install -r requirements.txt
```
weights will be downloaded from huggingface

debug:
when install densepose on windows, you may meet
```
安装densepose 提示文件扩展名过长 https://blog.csdn.net/weixin_43148691/article/details/117354062
```

## Tutorial
- [Demo]()

## WeChat Group
<div>
  <figure>
  <img alt='Wechat' src="wechat.jpg?raw=true" width="300px"/>
  <figure>
</div>

## Thanks
[ViViD](https://github.com/alibaba-yuanjing-aigclab/ViViD)
