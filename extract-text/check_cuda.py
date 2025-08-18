import paddle
print("CUDA compiled:", paddle.is_compiled_with_cuda())
paddle.set_device("gpu")
print("Device:", paddle.get_device())