# Vgg16-Tensorflow
This is an implement of vgg16 and vgg19 base on [machrisaa's](https://github.com/machrisaa/tensorflow-vgg) and [kratzert's](https://github.com/kratzert/finetune_alexnet_with_tensorflow) work. To use the VGG networks, the npy files for [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) or [VGG19 NPY](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) has to be downloaded. 

## Upgrade
The extra files(vgg16_trainable.py, datagenerator.py, test_vgg16_trainable.py and test_finetune_vgg16.py) can let you to apply the vgg16 network to others classification tasks through modifying the last fully-connected layer. Furthermore, you can easily choose any layers that you want to train instead of all layers. To train the vgg16 network, you need to prepare same datasets, or you can download one from [here](https://pan.baidu.com/s/1Drhq1Xrs5zTju690DHvf_Q).

The experiment environment is as follows:
- tensorflow=1.12.0
- pyhton=3.6.7
- numpy=1.16.2
