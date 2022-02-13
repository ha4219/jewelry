# Python 환경 구성
## Conda
```bash
# conda update
$ conda install -c anaconda cudatoolkit

$ conda activate face # 실행
$ conda deactivate # 중지

$ nvcc -V # cuda 버전 확인
$ lshw -c video # 그래픽 카드 확인
$ export TORCH_CUDA_ARCH_LIST=8.6 # 뒤에 숫자는 버전에 맞게 변경
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html # 이것도 버전 확인해서 다운로드 해야 됨
$ conda install pillow
$ pip install scikit-image
$ pip install tqdm
```

## Test Running
```bash
$ ~/anaconda3/envs/face/bin/python3.7 ~/jewelry/Face-parsing/run_dw.py ~/jewelry/test\ images/src000.jpg  ~/jewelry/Face-parsing/res/cp/79999_iter.pth ~/parsing/test.png
$ ~/anaconda3/envs/face/bin/python3.7 ~/jewelry/Face-alignment/run_dw.py ~/jewelry/test\ images/src000.jpg ~/alignment/test_1.bin
```

## Issue
1. Face-alignment의 face_alignmnet/api.py line 80의 경로를 절대값으로 해줘야 한다.
   1. 서버를 옮길 때 마다 바꿔줘야 하는데 상대 경로로 세팅하는 방법 없나..

# C++ 환경 구성
## cmake && opencv
```bash
$ sudo apt-get update && apt-get upgrade -y
$ sudo apt-get install -y software-properties-common
$ sudo add-apt-repository 'deb http://security.ubuntu.com/ubuntu xenial-security main'
$ sudo apt update && apt upgrade -y

$ sudo apt install -y build-essential cmake pkg-config git \
    libjasper1 libjpeg-dev libtiff5-dev libpng-dev libjasper-dev \
    libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxvidcore-dev libx264-dev libxine2-dev libv4l-dev v4l-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev libatlas-base-dev libeigen3-dev gfortran \
    python3-dev python3-numpy libtbb2 libtbb-dev
$ sudo apt-get install -y qt5-default qtbase5-dev qtdeclarative5-dev

# opencv 다운
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.7.zip && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.7.zip
```