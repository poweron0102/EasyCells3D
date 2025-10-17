import math
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32
import numpy as np


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
def quat_inverse(q):
    """Calcula o inverso de um quaternion (w, x, y, z)."""
    return q[0], -q[1], -q[2], -q[3]


@cuda.jit(device=True)
def quat_rotate_vector(q, v):
    """Rotaciona um vetor v por um quaternion q."""
    # Extrai as componentes do vetor e do quaternion
    vx, vy, vz = v
    qw, qx, qy, qz = q

    # Constrói o quaternion do vetor puro
    px, py, pz, pw = vx, vy, vz, 0.0

    # Realiza a multiplicação q * p
    res_x = qw * px + qx * pw + qy * pz - qz * py
    res_y = qw * py - qx * pz + qy * pw + qz * px
    res_z = qw * pz + qx * py - qy * px + qz * pw
    res_w = qw * pw - qx * px - qy * py - qz * pz

    # Extrai o conjugado de q
    conj_qw, conj_qx, conj_qy, conj_qz = qw, -qx, -qy, -qz

    # Realiza a multiplicação (q * p) * q_conj
    final_x = res_w * conj_qx + res_x * conj_qw + res_y * conj_qz - res_z * conj_qy
    final_y = res_w * conj_qy - res_x * conj_qz + res_y * conj_qw + res_z * conj_qx
    final_z = res_w * conj_qz + res_x * conj_qy - res_y * conj_qx + res_z * conj_qw

    return final_x, final_y, final_z


@cuda.jit(device=True)
def get_sphere_uv_gpu(world_normal, sphere_rotation):
    """Calcula as coordenadas UV para um ponto na esfera na GPU."""
    local_normal = quat_rotate_vector(quat_inverse(sphere_rotation), world_normal)
    theta = math.asin(local_normal[1])
    phi = math.atan2(local_normal[2], local_normal[0])
    u = 1.0 - (phi + math.pi) / (2.0 * math.pi)
    v = (theta + math.pi / 2.0) / math.pi
    return u, v

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


@cuda.jit(device=True)
def texture_lookup(texture, u, v):
    """Busca a cor de uma textura 2D."""
    # Inverte v porque a origem da textura (0,0) é geralmente no canto superior esquerdo,
    # mas o cálculo de UV assume que é no canto inferior esquerdo.
    v = 1.0 - v

    # Obtém as dimensões da textura
    height, width, _ = texture.shape

    # Converte coordenadas UV (0-1) para coordenadas de pixel
    x = int(u * (width - 1))
    y = int(v * (height - 1))

    # Garante que os índices estão dentro dos limites
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    # Busca a cor e normaliza para (0-1)
    r = texture[y, x, 0] / 255.0
    g = texture[y, x, 1] / 255.0
    b = texture[y, x, 2] / 255.0
    return r, g, b


@cuda.jit
def render_kernel(
        pixel_array,
        camera_center,
        pixel00_loc,
        pixel_delta_u,
        pixel_delta_v,
        spheres,
        materials,
        textures,
        light_direction,
        ambient_light
    ):
    """
    Kernel CUDA para renderizar a cena. Cada thread calcula a cor de um píxel.
    """
    # Calcula o índice do píxel (i, j) para esta thread específica
    i, j = cuda.grid(2)

    # Verifica se o píxel está dentro dos limites da imagem
    height, width, _ = pixel_array.shape
    if j >= height or i >= width:
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
    hit_point = (0.0, 0.0, 0.0)
    hit_material_index = -1
    hit_sphere_rotation = (1.0, 0.0, 0.0, 0.0)
    hit = False

    # Itera sobre todas as esferas na cena
    for s in range(len(spheres)):
        sphere_center = (spheres[s]['center'][0], spheres[s]['center'][1], spheres[s]['center'][2])
        sphere_radius = spheres[s]['radius']
        sphere_rotation = (spheres[s]['rotation'][0], spheres[s]['rotation'][1], spheres[s]['rotation'][2], spheres[s]['rotation'][3])

        dist, normal = sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius)

        if 0.001 < dist < min_dist:
            min_dist = dist
            hit_normal = normal
            hit_material_index = spheres[s]['material_index']
            hit_sphere_rotation = sphere_rotation
            hit = True

    # 3. Define a cor final do píxel
    if hit:
        # Iluminação Blinn-Phong
        mat = materials[hit_material_index]

        # Cor base (albedo)
        if mat['texture_index'] != -1:
            u, v = get_sphere_uv_gpu(hit_normal, hit_sphere_rotation)
            albedo_r, albedo_g, albedo_b = texture_lookup(
                textures[mat['texture_index']],
                u, v
            )
        else:
            albedo_r, albedo_g, albedo_b = mat['diffuse_color']

        # Componente emissiva
        emissive_r, emissive_g, emissive_b = mat['emissive_color']

        # Componente ambiente
        ambient_r = albedo_r * ambient_light[0]
        ambient_g = albedo_g * ambient_light[1]
        ambient_b = albedo_b * ambient_light[2]

        # Componente difusa
        diffuse_intensity = max(0.0, vec3_dot(hit_normal, light_direction))
        diffuse_r = albedo_r * diffuse_intensity
        diffuse_g = albedo_g * diffuse_intensity
        diffuse_b = albedo_b * diffuse_intensity

        # Componente especular
        view_dir = vec3_normalize((-ray_direction[0], -ray_direction[1], -ray_direction[2]))
        half_vector_x = light_direction[0] + view_dir[0]
        half_vector_y = light_direction[1] + view_dir[1]
        half_vector_z = light_direction[2] + view_dir[2]
        half_vector = vec3_normalize((half_vector_x, half_vector_y, half_vector_z))

        specular_intensity = pow(max(0.0, vec3_dot(hit_normal, half_vector)), mat['shininess'])
        specular_r = mat['specular'] * specular_intensity
        specular_g = mat['specular'] * specular_intensity
        specular_b = mat['specular'] * specular_intensity

        # Combina as componentes
        final_r = emissive_r + ambient_r + diffuse_r + specular_r
        final_g = emissive_g + ambient_g + diffuse_g + specular_g
        final_b = emissive_b + ambient_b + diffuse_b + specular_b

        # Converte para 8-bit e aplica clamp
        r = int(255.999 * min(1.0, final_r))
        g = int(255.999 * min(1.0, final_g))
        b = int(255.999 * min(1.0, final_b))

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
