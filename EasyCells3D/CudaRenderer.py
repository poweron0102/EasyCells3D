import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.driver import CompileError
from pycuda.compiler import SourceModule
import numpy as np
import os

class CudaRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixel_array = np.zeros((height, width, 4), dtype=np.uint8)

        # Carrega e compila o kernel CUDA
        kernel_path = os.path.join(os.path.dirname(__file__), 'render_kernel.cu')
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()

        print("Kernel code:\n", kernel_code, "\n")
        self.module = SourceModule(kernel_code, no_extern_c=True, keep=True)


        self.render_kernel = self.module.get_function("render_kernel")

        # Define a estrutura dos dados para corresponder ao kernel C++
        self.sphere_struct = np.dtype([
            ("center", np.float32, 3),
            ("radius", np.float32),
            ("material_index", np.int32),
            ("rotation", np.float32, 4) # w, x, y, z
        ])

        self.material_struct = np.dtype([
            ("diffuse_color", np.float32, 3),
            ("specular", np.float32),
            ("shininess", np.float32),
            ("texture_index", np.int32),
            ("emissive_color", np.float32, 3)
        ])

    def render(self, camera, scene, light_direction, ambient_light):
        # 1. Preparar dados para o kernel
        # Camera
        camera_center = np.array(camera.center, dtype=np.float32)
        pixel00_loc = np.array(camera.pixel00_loc, dtype=np.float32)
        pixel_delta_u = np.array(camera.pixel_delta_u, dtype=np.float32)
        pixel_delta_v = np.array(camera.pixel_delta_v, dtype=np.float32)

        # Esferas
        spheres_list = scene.get_hittables_of_type('SphereHittable')
        num_spheres = len(spheres_list)
        spheres_data = np.empty(num_spheres, dtype=self.sphere_struct)
        for i, sphere_comp in enumerate(spheres_list):
            spheres_data[i]['center'] = np.array(sphere_comp.game_object.transform.position, dtype=np.float32)
            spheres_data[i]['radius'] = sphere_comp.radius
            spheres_data[i]['material_index'] = scene.get_material_index(sphere_comp.material)
            spheres_data[i]['rotation'] = np.array(sphere_comp.game_object.transform.rotation, dtype=np.float32)


        # Materiais
        materials_list = scene.materials
        num_materials = len(materials_list)
        materials_data = np.empty(num_materials, dtype=self.material_struct)
        for i, mat in enumerate(materials_list):
            materials_data[i]['diffuse_color'] = np.array(mat.diffuse_color, dtype=np.float32)
            materials_data[i]['specular'] = mat.specular
            materials_data[i]['shininess'] = mat.shininess
            materials_data[i]['texture_index'] = scene.get_texture_index(mat.texture) if mat.texture else -1
            materials_data[i]['emissive_color'] = np.array(mat.emissive_color, dtype=np.float32)

        # Texturas
        textures_list = scene.textures
        if textures_list:
            # Concatena todos os dados de textura em um único buffer
            texture_buffers = [tex.data.astype(np.uint8).flatten() for tex in textures_list]
            textures_data_gpu = np.concatenate(texture_buffers)
            
            # Cria um array de informações sobre cada textura (offset, width, height)
            texture_info = np.empty(len(textures_list), dtype=np.int32, order='C')
            texture_info = texture_info.reshape(-1, 3) # Garante que é Nx3
            current_offset = 0
            for i, tex in enumerate(textures_list):
                height, width, _ = tex.data.shape
                texture_info[i] = [current_offset, width, height]
                current_offset += width * height * 3 # 3 canais
        else:
            # Arrays vazios se não houver texturas
            textures_data_gpu = np.empty(0, dtype=np.uint8)
            texture_info = np.empty((0, 3), dtype=np.int32)


        # Luzes
        light_direction_gpu = np.array(light_direction, dtype=np.float32)
        ambient_light_gpu = np.array(ambient_light, dtype=np.float32)

        # 2. Alocar memória na GPU e copiar dados
        pixel_array_gpu = cuda.mem_alloc(self.pixel_array.nbytes)
        
        spheres_gpu = cuda.mem_alloc(spheres_data.nbytes)
        cuda.memcpy_htod(spheres_gpu, spheres_data)

        materials_gpu = cuda.mem_alloc(materials_data.nbytes)
        cuda.memcpy_htod(materials_gpu, materials_data)
        
        textures_data_gpu_ptr = cuda.mem_alloc(textures_data_gpu.nbytes)
        cuda.memcpy_htod(textures_data_gpu_ptr, textures_data_gpu)
        
        texture_info_gpu_ptr = cuda.mem_alloc(texture_info.nbytes)
        cuda.memcpy_htod(texture_info_gpu_ptr, texture_info)


        # 3. Lançar o kernel
        block_size = (16, 16, 1)
        grid_size = (
            (self.width + block_size[0] - 1) // block_size[0],
            (self.height + block_size[1] - 1) // block_size[1]
        )

        self.render_kernel(
            pixel_array_gpu,
            np.int32(self.width), np.int32(self.height),
            cuda.In(camera_center),
            cuda.In(pixel00_loc),
            cuda.In(pixel_delta_u),
            cuda.In(pixel_delta_v),
            spheres_gpu, np.int32(num_spheres),
            materials_gpu, np.int32(num_materials),
            textures_data_gpu_ptr,
            texture_info_gpu_ptr,
            cuda.In(light_direction_gpu),
            cuda.In(ambient_light_gpu),
            block=block_size,
            grid=grid_size
        )

        # 4. Copiar o resultado de volta para a CPU
        cuda.memcpy_dtoh(self.pixel_array, pixel_array_gpu)
        
        # Liberar memória da GPU
        pixel_array_gpu.free()
        spheres_gpu.free()
        materials_gpu.free()
        textures_data_gpu_ptr.free()
        texture_info_gpu_ptr.free()


        return self.pixel_array
