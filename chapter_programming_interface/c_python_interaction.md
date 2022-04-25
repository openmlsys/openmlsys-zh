## C/C++编程接口

在上述小节中，我们讨论了开发者如何利用Python来定义机器学习的整个工作流，以及如何定义复杂的深度神经网络。然而，在很多时候，用户也需要添加自定义的算子来帮助实现新的模型，优化器，数据处理函数等。这些自定义算子需要通过C和C++实现，从而获得最优性能。但是为了帮助这些算子被用户使用，他们也需要暴露为Python函数，从而方便用户整合入已有的Python为核心编写的工作流和模型。在这一小节中，我们讨论这一过程是如何实现的。

### 在Python中调用C/C++函数的原理

由于Python的解释器是由C实现的，因此在Python中可以实现对于C和C++函数的调用。现代机器学习框架（包括TensorFlow，PyTorch和MindSpore）主要依赖Pybind11来将底层的大量C和C++函数自动生成对应的Python函数，这一过程一般被称为Python绑定（
Binding）。在Pybind11出现以前，将C和C++函数进行Python绑定的手段主要包括：

- Python的C-API。这种方式要求在一个C++程序中包含Python.h，并使用Python的C-API对Python语言进行操作。使用这套API需要对Python的底层实现有一定了解，比如如何管理引用计数等，具有较高的使用门槛。

- 简单包装界面产生器（Simplified Wrapper and Interface Generator，SWIG)。SWIG可以将C和C++代码暴露给Python。SWIG是TensorFlow早期使用的方式。这种方式需要用户编写一个复杂的SWIG接口声明文件，并使用SWIG自动生成使用Python
    C-API的C代码。自动生成的代码可读性很低，因此具有很大代码维护开销。

-  Python的ctypes模块，提供了C语言中的类型，以及直接调用动态链接库的能力。缺点是依赖于C的原生的类型，对自定义类型支持不好。

- Cython是结合了Python和C语言的一种语言，可以简单的认为就是给Python加上了静态类型后的语法，使用者可以维持大部分的Python语法。Cython编写的函数会被自动转译为C和C++代码，因此在Cython中可以插入对于C/C++函数的调用。

- Boost::Python是一个C++库。它可以将C++函数暴露为Python函数。其原理和Python C-API类似，但是使用方法更简单。然而，由于引入了Boost库，因此有沉重的第三方依赖。

相对于上述的提供Python绑定的手段，Pybind11提供了类似于Boost::Python的简洁性和易用性，但是其通过专注支持C++
11，并且去除Boost依赖，因此成为了轻量级的Python库，从而特别适合在一个复杂的C++项目（例如本书讨论的机器学习系统）中暴露大量的Python函数。

### 添加C++编写的自定义算子

算子是构建神经网络的基础，在前面也称为低级API；通过算子的封装可以实现各类神经网络层，当开发神经网络层遇到内置算子无法满足时，可以通过自定义算子来实现。以MindSpore为例，实现一个GPU算子需要如下步骤：

1.  Primitive注册：算子原语是构建网络模型的基础单元，用户可以直接或者间接调用算子原语搭建一个神经网络模型。

2.  GPU Kernel实现：GPU Kernel用于调用GPU实现加速计算。

3.  GPU Kernel注册：算子注册用于将GPU
    Kernel及必要信息注册给框架，由框架完成对GPU Kernel的调用。

**1.注册算子原语**
算子原语通常包括算子名、算子输入、算子属性（初始化时需要填的参数，如卷积的stride、padding）、输入数据合法性校验、输出数据类型推导和维度推导。假设需要编写加法算子，主要内容如下：

-   算子名：TensorAdd

-   算子属性：构造函数\_\_init\_\_中初始化属性，因加法没有属性，因此\_\_init\_\_不需要额外输入。

-   算子输入输出及合法性校验：infer_shape方法中约束两个输入维度必须相同，输出的维度和输入维度相同。infer_dtype方法中约束两个输入数据必须是float32类型，输出的数据类型和输入数据类型相同。

-   算子输出

MindSpore中实现注册TensorAdd代码如下：
```python
# mindspore/ops/operations/math_ops.py
class TensorAdd(PrimitiveWithInfer):
    """
    Adds two input tensors element-wise.
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])

    def infer_shape(self, x1_shape, x2_shape):
        validator.check_integer('input dims', len(x1_shape), len(x2_shape), Rel.EQ, self.name)
        for i in range(len(x1_shape)):
            validator.check_integer('input_shape', x1_shape[i], x2_shape[i], Rel.EQ, self.name)
        return x1_shape

    def infer_dtype(self, x1_dtype, x2_type):
        validator.check_tensor_type_same({'x1_dtype': x1_dtype}, [mstype.float32], self.name)
        validator.check_tensor_type_same({'x2_dtype': x2_dtype}, [mstype.float32], self.name)
        return x1_dtype
```
    
在mindspore/ops/operations/math_ops.py文件内注册加法算子原语后，需要在mindspore/ops/operations/\_\_init\_\_中导出，方便python导入模块时候调用。
```python
# mindspore/ops/operations/__init__.py
from .math_ops import (Abs, ACos, ..., TensorAdd)
__all__ = [
  'ReverseSequence',
  'CropAndResize',
  ...,
  'TensorAdd'
]
```

**2.GPU算子开发**继承GPUKernel，实现加法使用类模板定义TensorAddGpuKernel，需要实现以下方法：

- Init(): 用于完成GPU Kernel的初始化，通常包括记录算子输入/输出维度，完成Launch前的准备工作；因此在此记录Tensor元素个数。

- GetInputSizeList():向框架反馈输入Tensor需要占用的显存字节数；返回了输入Tensor需要占用的字节数，TensorAdd有两个Input，每个Input占用字节数为element_num$\ast$sizeof(T)。

- GetOutputSizeList():向框架反馈输出Tensor需要占用的显存字节数；返回了输出Tensor需要占用的字节数，TensorAdd有一个output，占用element_num$\ast$sizeof(T)字节。

- GetWorkspaceSizeList():向框架反馈Workspace字节数，Workspace是用于计算过程中存放临时数据的空间；由于TensorAdd不需要Workspace，因此GetWorkspaceSizeList()返回空的std::vector\<size_t\>。

- Launch(): 通常调用CUDA kernel(CUDA kernel是基于Nvidia GPU的并行计算架构开发的核函数)，或者cuDNN接口等方式，完成算子在GPU上加速；Launch()接收input、output在显存的地址，接着调用TensorAdd完成加速。
```python
// mindspore/ccsrc/backend/kernel_compiler/gpu/math/tensor_add_v2_gpu_kernel.h

template <typename T>
class TensorAddGpuKernel : public GpuKernel {
 public:
  TensorAddGpuKernel() : element_num_(1) {}
  ~TensorAddGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < shape.size(); i++) {
      element_num_ *= shape[i];
    }
    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *x1 = GetDeviceAddress<T>(inputs, 0);
    T *x2 = GetDeviceAddress<T>(inputs, 1);
    T *y = GetDeviceAddress<T>(outputs, 0);

    TensorAdd(element_num_, x1, x2, y, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(element_num_ * sizeof(T));
    input_size_list_.push_back(element_num_ * sizeof(T));
    output_size_list_.push_back(element_num_ * sizeof(T));
  }

 private:
  size_t element_num_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
```

TensorAdd中调用了CUDA
kernelTensorAddKernel来实现element_num个元素的并行相加:
```python
// mindspore/ccsrc/backend/kernel_compiler/gpu/math/tensor_add_v2_gpu_kernel.h

 template <typename T>
 __global__ void TensorAddKernel(const size_t element_num, const T* x1, const T* x2, T* y) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < element_num; i += blockDim.x * gridDim.x) {
    y[i] = x1[i] + x2[i];
  }
 }

 template <typename T>
 void TensorAdd(const size_t &element_num, const T* x1, const T* x2, T* y, cudaStream_t stream){
    size_t thread_per_block = 256;
    size_t block_per_grid = (element_num + thread_per_block - 1 ) / thread_per_block;
    TensorAddKernel<<<block_per_grid, thread_per_block, 0, stream>>>(element_num, x1, x2, y);
   return;
 }

 template void TensorAdd(const size_t &element_num, const float* x1, const float* x2, float* y, cudaStream_t stream);
```

**3.GPU算子注册**算子信息包含1.Primive；2.Input dtype, output dtype；3.GPU Kernel class；
4.CUDA内置数据类型。框架会根据Primive和Input dtype, output dtype，调用以CUDA内置数据类型实例化GPU Kernel class模板类。如下代码中分别注册了支持float和int的TensorAdd算子。
```python
// mindspore/ccsrc/backend/kernel_compiler/gpu/math/tensor_add_v2_gpu_kernel.cc

MS_REG_GPU_KERNEL_ONE(TensorAddV2, KernelAttr()
                                    .AddInputAttr(kNumberTypeFloat32)
                                    .AddInputAttr(kNumberTypeFloat32)
                                    .AddOutputAttr(kNumberTypeFloat32),
                      TensorAddV2GpuKernel, float)

MS_REG_GPU_KERNEL_ONE(TensorAddV2, KernelAttr()
                                    .AddInputAttr(kNumberTypeInt32)
                                    .AddInputAttr(kNumberTypeInt32)
                                    .AddOutputAttr(kNumberTypeInt32),
                      TensorAddV2GpuKernel, int)
```
    
完成上述三步工作后，需要把MindSpore重新编译，在源码的根目录执行bash
build.sh -e gpu，最后使用算子进行验证。
