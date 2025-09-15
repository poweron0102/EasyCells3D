import math
from numba import cuda
import numpy as np


# ==================================================================================================
# Funções de Dispositivo (Executam na GPU para cada thread)
# ==================================================================================================
# Estas são funções auxiliares que são chamadas a partir do kernel principal.
# O decorador `cuda.jit(device=True)` indica que elas serão compiladas para código de GPU.

@cuda.jit(device=True)
def vec3_dot(a, b):
    """Calcula o produto escalar de dois vetores 3D."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@cuda.jit(device=True)
def vec3_length_squared(a):
    """Calcula o quadrado da magnitude de um vetor 3D."""
    return a[0] ** 2 + a[1] ** 2 + a[2] ** 2


@cuda.jit(device=True)
def vec3_normalize(a):
    """Normaliza um vetor 3D."""
    len_sq = vec3_length_squared(a)
    if len_sq > 0:
        inv_len = 1.0 / math.sqrt(len_sq)
        return a[0] * inv_len, a[1] * inv_len, a[2] * inv_len
    return a


@cuda.jit(device=True)
def sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius):
    """
    Calcula a interseção de um raio com uma esfera.
    Retorna a distância da interseção ou um valor negativo se não houver acerto.
    """
    oc_x = ray_origin[0] - sphere_center[0]
    oc_y = ray_origin[1] - sphere_center[1]
    oc_z = ray_origin[2] - sphere_center[2]

    a = vec3_dot(ray_direction, ray_direction)
    half_b = oc_x * ray_direction[0] + oc_y * ray_direction[1] + oc_z * ray_direction[2]
    c = oc_x ** 2 + oc_y ** 2 + oc_z ** 2 - sphere_radius ** 2
    discriminant = half_b * half_b - a * c

    if discriminant < 0:
        return -1.0, (0.0, 0.0, 0.0)  # Sem acerto

    sqrt_d = math.sqrt(discriminant)
    root = (-half_b - sqrt_d) / a
    if root <= 0.001:
        root = (-half_b + sqrt_d) / a
        if root <= 0.001:
            return -1.0, (0.0, 0.0, 0.0)  # Sem acerto

    # Calcula o ponto de acerto e a normal
    hit_point_x = ray_origin[0] + root * ray_direction[0]
    hit_point_y = ray_origin[1] + root * ray_direction[1]
    hit_point_z = ray_origin[2] + root * ray_direction[2]

    normal_x = hit_point_x - sphere_center[0]
    normal_y = hit_point_y - sphere_center[1]
    normal_z = hit_point_z - sphere_center[2]

    normal = vec3_normalize((normal_x, normal_y, normal_z))

    return root, normal


# ==================================================================================================
# Kernel CUDA (Executa na GPU)
# ==================================================================================================
# Esta é a função principal que a GPU irá executar.
# Cada thread da GPU executará esta função para um píxel diferente.

@cuda.jit
def render_kernel(pixel_array, camera_center, pixel00_loc, pixel_delta_u, pixel_delta_v, spheres):
    """
    Kernel CUDA para renderizar a cena. Cada thread calcula a cor de um píxel.
    """
    # Calcula o índice do píxel (i, j) para esta thread específica
    i, j = cuda.grid(2)

    # Verifica se o píxel está dentro dos limites da imagem
    height, width, _ = pixel_array.shape
    if i >= width or j >= height:
        return

    # 1. Calcula o raio para o píxel atual (equivalente a `_get_ray`)
    pixel_center_x = pixel00_loc[0] + i * pixel_delta_u[0] + j * pixel_delta_v[0]
    pixel_center_y = pixel00_loc[1] + i * pixel_delta_u[1] + j * pixel_delta_v[1]
    pixel_center_z = pixel00_loc[2] + i * pixel_delta_u[2] + j * pixel_delta_v[2]

    ray_dir_x = pixel_center_x - camera_center[0]
    ray_dir_y = pixel_center_y - camera_center[1]
    ray_dir_z = pixel_center_z - camera_center[2]

    ray_direction = vec3_normalize((ray_dir_x, ray_dir_y, ray_dir_z))
    ray_origin = camera_center

    # 2. Calcula a cor do raio (equivalente a `_ray_color`)
    min_dist = 1e10  # Um número muito grande (float('inf') não é suportado no modo device)
    hit_normal = (0.0, 0.0, 0.0)
    hit = False

    # Itera sobre todas as esferas na cena
    for s in range(len(spheres)):
        sphere_center = (spheres[s]['center'][0], spheres[s]['center'][1], spheres[s]['center'][2])
        sphere_radius = spheres[s]['radius']

        dist, normal = sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius)

        if dist > 0.001 and dist < min_dist:
            min_dist = dist
            hit_normal = normal
            hit = True

    # 3. Define a cor final do píxel
    if hit:
        # Colore com base na normal da superfície
        r = int(255.999 * (hit_normal[0] + 1) * 0.5)
        g = int(255.999 * (hit_normal[1] + 1) * 0.5)
        b = int(255.999 * (hit_normal[2] + 1) * 0.5)
    else:
        # Cor de fundo (gradiente do céu)
        a = 0.5 * (ray_direction[1] + 1.0)
        r = int(255.999 * ((1.0 - a) * 1.0 + a * 0.5))
        g = int(255.999 * ((1.0 - a) * 1.0 + a * 0.7))
        b = int(255.999 * ((1.0 - a) * 1.0 + a * 1.0))

    # Escreve a cor no array de píxeis
    pixel_array[j, i, 0] = r
    pixel_array[j, i, 1] = g
    pixel_array[j, i, 2] = b
