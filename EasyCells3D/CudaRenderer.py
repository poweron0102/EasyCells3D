import math
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# --- Taichi Data Types ---
Vec3f = ti.types.vector(3, ti.f32)
Quat = ti.types.vector(4, ti.f32)


# --- Taichi Functions (run on GPU) ---

@ti.func
def quat_inverse(q: Quat) -> Quat:
    """Calcula o inverso de um quaternion (w, x, y, z)."""
    return Quat(q[0], -q[1], -q[2], -q[3])


@ti.func
def quat_rotate_vector(q: Quat, v: Vec3f) -> Vec3f:
    """Rotaciona um vetor v por um quaternion q."""
    # Converte o vetor para um quaternion puro
    p = Quat(0.0, v.x, v.y, v.z)

    # q * p
    res_w = q.w * p.w - q.x * p.x - q.y * p.y - q.z * p.z
    res_x = q.w * p.x + q.x * p.w + q.y * p.z - q.z * p.y
    res_y = q.w * p.y - q.x * p.z + q.y * p.w + q.z * p.x
    res_z = q.w * p.z + q.x * p.y - q.y * p.x + q.z * p.w
    qp = Quat(res_w, res_x, res_y, res_z)

    # (q * p) * q_conj
    q_conj = quat_inverse(q)
    final_w = qp.w * q_conj.w - qp.x * q_conj.x - qp.y * q_conj.y - qp.z * q_conj.z
    final_x = qp.w * q_conj.x + qp.x * q_conj.w + qp.y * q_conj.z - qp.z * q_conj.y
    final_y = qp.w * q_conj.y - qp.x * q_conj.z + qp.y * q_conj.w + qp.z * q_conj.x
    final_z = qp.w * q_conj.z + qp.x * q_conj.y - qp.y * q_conj.x + qp.z * q_conj.w

    return Vec3f(final_x, final_y, final_z)


@ti.func
def get_sphere_uv(world_normal: Vec3f, sphere_rotation: Quat) -> ti.types.vector(2, ti.f32):
    """Calcula as coordenadas UV para um ponto na esfera."""
    local_normal = quat_rotate_vector(quat_inverse(sphere_rotation), world_normal)
    theta = ti.asin(local_normal.y)
    phi = ti.atan2(local_normal.z, local_normal.x)
    u = 1.0 - (phi + math.pi) / (2.0 * math.pi)
    v = (theta + math.pi / 2.0) / math.pi
    return ti.Vector([u, v])


@ti.func
def sphere_intersect(ray_origin: Vec3f, ray_direction: Vec3f, sphere_center: Vec3f, sphere_radius: ti.f32):
    """
    Calcula a interseção de um raio com uma esfera.
    Retorna a distância, a normal do acerto e um booleano de sucesso.
    """
    oc = ray_origin - sphere_center
    a = ray_direction.dot(ray_direction)
    half_b = oc.dot(ray_direction)
    c = oc.dot(oc) - sphere_radius * sphere_radius
    discriminant = half_b * half_b - a * c

    hit = False
    root = -1.0
    normal = Vec3f(0.0)

    if discriminant >= 0:
        sqrt_d = ti.sqrt(discriminant)
        temp_root = (-half_b - sqrt_d) / a
        if temp_root > 0.001:
            root = temp_root
            hit = True
        else:
            temp_root = (-half_b + sqrt_d) / a
            if temp_root > 0.001:
                root = temp_root
                hit = True

    if hit:
        hit_point = ray_origin + root * ray_direction
        normal = (hit_point - sphere_center).normalized()

    return root, normal, hit


@ti.func
def texture_lookup(texture: ti.template(), u: ti.f32, v: ti.f32) -> Vec3f:
    """Busca a cor de uma textura 2D."""
    v = 1.0 - v  # Inverte v para corresponder ao sistema de coordenadas de textura
    height, width = texture.shape[0], texture.shape[1]

    # Converte coordenadas UV (0-1) para coordenadas de pixel
    x = ti.cast(u * (width - 1), ti.i32)
    y = ti.cast(v * (height - 1), ti.i32)

    # Garante que os índices estão dentro dos limites
    x = ti.max(0, ti.min(width - 1, x))
    y = ti.max(0, ti.min(height - 1, y))

    # Busca a cor e normaliza para (0-1)
    return Vec3f(
        texture[y, x, 0] / 255.0,
        texture[y, x, 1] / 255.0,
        texture[y, x, 2] / 255.0
    )


@ti.kernel
def render_kernel(
        pixels: ti.template(),
        camera_center: Vec3f,
        pixel00_loc: Vec3f,
        pixel_delta_u: Vec3f,
        pixel_delta_v: Vec3f,
        spheres: ti.template(),
        materials: ti.template(),
        texture: ti.template(),
        light_direction: Vec3f,
        ambient_light: Vec3f
):
    """
    Kernel Taichi para renderizar a cena. Cada thread calcula a cor de um píxel.
    """
    for i, j in pixels:  # Loop paralelo sobre todos os pixels da imagem
        # 1. Calcula o raio para o píxel atual
        pixel_center = pixel00_loc + i * pixel_delta_u + j * pixel_delta_v
        ray_direction = (pixel_center - camera_center).normalized()
        ray_origin = camera_center

        # 2. Encontra a interseção mais próxima
        min_dist = 1e10
        hit_normal = Vec3f(0.0)
        hit_material_index = -1
        hit_sphere_rotation = Quat(1.0, 0.0, 0.0, 0.0)
        any_hit = False

        for s in range(spheres.shape[0]):
            dist, normal, hit_success = sphere_intersect(
                ray_origin,
                ray_direction,
                spheres[s].center,
                spheres[s].radius
            )

            if hit_success and dist < min_dist:
                min_dist = dist
                hit_normal = normal
                hit_material_index = spheres[s].material_index
                hit_sphere_rotation = spheres[s].rotation
                any_hit = True

        # 3. Calcula a cor final do píxel
        final_color_rgb = Vec3f(0.0)
        if any_hit:
            mat = materials[hit_material_index]
            albedo = mat.diffuse_color

            # Cor base (albedo) da textura, se houver
            if mat.texture_index != -1:
                uv = get_sphere_uv(hit_normal, hit_sphere_rotation)
                albedo = texture_lookup(texture, uv.x, uv.y)

            # Iluminação Blinn-Phong
            emissive = mat.emissive_color
            ambient = albedo * ambient_light
            diffuse_intensity = ti.max(0.0, hit_normal.dot(light_direction))
            diffuse = albedo * diffuse_intensity

            view_dir = -ray_direction
            half_vector = (light_direction + view_dir).normalized()
            specular_intensity = ti.pow(ti.max(0.0, hit_normal.dot(half_vector)), mat.shininess)
            specular = mat.specular * specular_intensity * Vec3f(1.0)  # Cor especular branca

            final_color_rgb = emissive + ambient + diffuse + specular
            final_color_rgb = ti.min(Vec3f(1.0), final_color_rgb) # Clamp
        else:
            # Cor de fundo (gradiente do céu)
            a = 0.5 * (ray_direction.y + 1.0)
            final_color_rgb = (1.0 - a) * Vec3f(1.0, 1.0, 1.0) + a * Vec3f(0.5, 0.7, 1.0)

        # Converte de float (0-1) para u8 (0-255) e escreve no píxel
        pixels[i, j] = ti.cast(255.999 * final_color_rgb, ti.u8)
