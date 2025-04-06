# torch_knn

## Reference:
[KNN_CUDA](https://github.com/unlimblue/KNN_CUDA),
[torchKNN](https://github.com/foolyc/torchKNN),
[DenseFusion-1](https://github.com/drapado/DenseFusion-1/tree/Pytorch-1.6),


以上代码在ubuntu20.04 cuda-12.1 Pytorch-2.4.0下测试成功
## Installation for densefusion 
```python
cd torch_knn/
cp -r knn  YOUR_DenseFusion_PATH/DenseFusion/lib/
cd YOUR_DenseFusion_PATH/DenseFusion/lib/knn
make
make clean
```
### If you want uninstall this package,please use:
```python
make uninstall
```
