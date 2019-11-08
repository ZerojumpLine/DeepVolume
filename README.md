# DeepVolume

## Test
- Download the example data in [Data](https://drive.google.com/file/d/1D9kZRk9p5f7KD2ZHgItzjRg5bP1wiOrp/view?usp=sharing). 
- Unzip the data and put them in folder dataForExamples
- Download the pretrain model in [PretrainModel](https://drive.google.com/file/d/1Eyhnj9kyXllOayW3YC64MuQo58zT9hf7/view?usp=sharing).
- Unzip the model and put them in folder models
- cd test & run DeepVolume_test.py -s 1
- cd test & run DeepVolume_test.py -s 2
- Find the results as ./output/test1/pred.nii.gz

## Train
As I cannot share all the data for public use, the training cannot be reproduced totally. However I provide all the code for training, it might be helpful for related applications.

- Download the example data in [Data](https://drive.google.com/file/d/1D9kZRk9p5f7KD2ZHgItzjRg5bP1wiOrp/view?usp=sharing). 
- Unzip the data and put them in folder dataForExamples
- Download the pretrain model in [PretrainModel](https://drive.google.com/file/d/1Eyhnj9kyXllOayW3YC64MuQo58zT9hf7/view?usp=sharing).
- Unzip the model and put them in folder models
- cd preprocessing & run SamplingforBrainStructureAwareNetwork.m
- cd train & run BrainStructureAwareNetwork_train.py
- cd preprocessing & run SamplingForSpatialConnectionAwareNetwork.m
- cd train & run SpatialConnectionAwareNetwork_train.py

## Others
In the paper, we also do some preprocessing and analyses based on SPM. The codes are trivial. However, if you are interested in that or have some problems running codes in this repository, please contact me through email (zeju.li18@imperial.ac.uk).

If you are interested in the full access of the data in the paper, please contact Prof. Yu for details. (jhyu@fudan.edu.cn)

## Citation
@article{li2019deepvolume,
  title={DeepVolume: Brain Structure and Spatial Connection-Aware Network for Brain MRI Super-Resolution},
  author={Li, Zeju and Yu, Jinhua and Wang, Yuanyuan and Zhou, Hanzhang and Yang, Haowei and Qiao, Zhongwei},
  journal={IEEE transactions on cybernetics},
  year={2019},
  publisher={IEEE}
}
