# import os
#
# from pycuda.compiler import SourceModule
#
# kernel_path = os.path.join(os.path.dirname(__file__), 'render_kernel.cu')
# with open(kernel_path, 'r') as f:
#     kernel_code = f.read()
#
# #print("Kernel code:\n", kernel_code, "\n")
# try:
#     module = SourceModule(kernel_code, no_extern_c=True, keep=True)
# except Exception as e:
#     print(e)
#     raise
#
# print(module)







import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# CUDA kernel code
mod = SourceModule(
"""
    __global__ void add_them(float *dest, float *a, float *b)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        dest[i] = a[i] + b[i];
    }
""")

# Get the kernel function
add_them = mod.get_function("add_them")
