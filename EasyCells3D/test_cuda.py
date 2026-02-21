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

import numpy

# Cria dados de exemplo
n = 400
a = numpy.random.randn(n).astype(numpy.float32)
b = numpy.random.randn(n).astype(numpy.float32)

# Aloca memória no dispositivo (GPU)
dest = numpy.zeros_like(a)
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
dest_gpu = cuda.mem_alloc(dest.nbytes)

# Copia os dados para o dispositivo
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Define o tamanho do bloco e do grid
block_size = 256
grid_size = (n + block_size - 1) // block_size

# Chama o kernel
add_them(dest_gpu, a_gpu, b_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

# Copia o resultado de volta para o host (CPU)
cuda.memcpy_dtoh(dest, dest_gpu)

# Verifica se o resultado está correto
if numpy.allclose(dest, a + b):
    print("Teste do PyCUDA bem-sucedido!")
    print("Resultado da GPU corresponde ao resultado da CPU.")
else:
    print("Teste do PyCUDA falhou!")









print("====" * 20)


import os

kernel_path = os.path.join(os.path.dirname(__file__), 'render_kernel.cu')
with open(kernel_path, 'r') as f:
    kernel_code = f.read()

#print("Kernel code:\n", kernel_code, "\n")
try:
    #module = SourceModule(kernel_code, no_extern_c=True, keep=False)
    a = cuda.to_device(b"Nathan e legal!\n")
    print("Type of array on device: ", type(a))
except Exception as e:
    print(e)
    raise

#print(module)


