# FCHD-Fully-Convolutional-Head-Detector
Code for FCHD - A fast and accurate head detector

This is the code for FCHD - A Fast and accurate head detector. See [the paper](https://arxiv.org/abs/1809.08766) for details and [video](https://youtu.be/gRPA7Hqk3VQ) for demo.

## Dependencies
- The code is tested on Ubuntu 16.04.

- install PyTorch >=0.4 with GPU (code are GPU-only), refer to [official website](http://pytorch.org)

- install cupy, you can install via `pip install cupy-cuda80` or(cupy-cuda90,cupy-cuda91, etc).

- install visdom for visualization, refer to their [github page](https://github.com/facebookresearch/visdom)

## Installation
1) Install Pytorch

2) Clone this repository
  ```Shell
  git clone https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector
  ```
3) Build cython code for speed:
  ```Bash
  cd src/nms/
  python build.py build_ext --inplace
  ```

## Training
1) Download the caffe pre-trained VGG16 from the following [link](https://drive.google.com/open?id=10AwNitG-5gq-YEJcG9iihosiOu7vAnfO). Store this pre-trained model in `data/pretrained_model ` folder.

2) Download the BRAINWASH dataset from the [official website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/software-and-datasets/). Unzip it and store the dataset in the `data/ ` folder.

3) Make appropriate settings in `src/config.py ` file regarding the updated paths, if required. The default paths set in `src/config.py ` are:
```
brainwash_dataset_root_path = 'data/brainwash_raw'
hollywood_dataset_root_path = 'data/HollywoodHeads'
caffe_pretrain_path = 'data/pretrained_model/vgg16_caffe.pth'
```
All paths are relative to the root directory. You can put the aforementioned files under these paths and use the model as-is without changing anything.

4) Start visdom server for visualization:
```Bash
python -m visdom.server
```
5) Run the following command to train the model: `python train.py `.

## Demo
1) Download the best performing model from the following [link](https://drive.google.com/open?id=1DbE4tAkaFYOEItwuIQhlbZypuIPDrArM).

2) Store the head detection model in `checkpoints/ ` folder.

3) Download the caffe pre-trained VGG16 from the following [link](https://drive.google.com/open?id=10AwNitG-5gq-YEJcG9iihosiOu7vAnfO). Store this pre-trained model in `data/pretrained_model ` folder.

4) Start visdom server for visualization.:
```Bash
python -m visdom.server
```

4) Run the following python command from the root folder.
```Shell
python head_detection_demo.py --img_path <test_image_path> --model_path <model_path>
```
_You can drop the `--model_path ` argument if  you have stored the head detection model under `checkpoints/ `._

5) The output of the model will be stored in a directory named `output/ ` in the same folder.

## Results
|              Method              |     AP     |
| :--------------------------------------: | :---------: |
| Overfeat - AlexNet [1] |    0.62    |
|   ReInspect, Lfix [1]    | 0.60 |
| ReInspect, Lfirstk [1]  | 0.63 |
| ReInspect, Lhungarian [1] | 0.78 |
| **Ours** | **0.70** |

## Runtime
- Runs at 5fps on NVidia Quadro M1000M GPU with 512 CUDA cores.

## Acknowledgement
This work builds on many of the excellent works:
- [Simple faster rcnn pytorch implementation](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) by [Yun Chen](https://github.com/chenyuntc).
- [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) by [Ross Girshick](https://github.com/rbgirshick).


## Reference
[1] Stewart, Russell, Mykhaylo Andriluka, and Andrew Y. Ng. "End-to-end people detection in crowded scenes." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

## MOBI

https://stackoverflow.com/questions/65695724/error-pythreadstate-aka-struct-ts-has-no-member-named-exc-type-did-yo

https://askubuntu.com/questions/883109/fatal-error-numpy-arrayobject-h-no-such-file-or-directory

https://www.datasciencelearner.com/assertionerror-torch-not-compiled-with-cuda-enabled-fix/

-----

# utils
cuda version: 
> nvidia-smi
> 
> conda activate mobi_venv
> 
pytorch installation with cuda support
https://pytorch.org/get-started/locally/

#package install
> conda activate mobi_venv
> 
> conda install -c conda-forge cupy
> 
> pip install scikit-image
> 
> pip install torchnet
> 
> pip install opencv-python
> 
> pip install watchdog
> 
> conda install pytorch==1.12.1 torchvision==1.12.1 torchaudio==1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# start webcam display and saving files for processing
>python webcam.py

# start head detection
>python head_detection_demo.py --img_path input/2.jpg

# display head detection in separate window
>python display.py --watch_file output/2.png

# files
https://drive.google.com/drive/folders/1cfVv0LAhvrkphMktuIWiDjx1dZ3XrvcZ?usp=share_link
