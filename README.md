# torch_knn

## Reference:
[KNN_CUDA](https://github.com/unlimblue/KNN_CUDA),
[torchKNN](https://github.com/foolyc/torchKNN),
[DenseFusion-1](https://github.com/drapado/DenseFusion-1/tree/Pytorch-1.6),


以上代码在ubuntu20.04 cuda-12.1 Pytorch-2.4.0下测试成功
## Installation for DenseFusion 
```python
cd torch_knn/
cp -r knn  YOUR_DenseFusion_PATH/DenseFusion/lib/
cd YOUR_DenseFusion_PATH/DenseFusion/lib/knn
make
make clean
```


## NOTE:First You need to install the **pytorch** and **CUDA-toolkit**
### If you want  to test the package in other environment, such as **CUDA-toolkit>=11.0** and **Pytorch>=1.8**,you need to execute these command line:

```python
conda activate $YOUR_conda_ENV
cd  torch_knn/knn
make
cp  -r build/lib.linux-x86_64*/* .
make clean
rm -rf torch_knn/torch_knn.cpython-39-x86_64-linux-gnu.so
```
### If you want uninstall this package,please use:
```python
make uninstall
```
