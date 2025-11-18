#include <math.h>



// ===================================================================
// 1. DEFINIÇÕES DAS STRUCTS
// (Isto é o que estava no .h - define o layout da memória)
// ===================================================================

typedef struct {
    float x;
    float y;
} Vec2f;

typedef struct {
    float x;
    float y;
    float z;
} Vec3f;

typedef struct {
    float w;
    float x;
    float y;
    float z;
} Quaternion;

typedef struct {
    Vec3f origin;
    Vec3f direction;
} Ray;


// ===================================================================
// 2. FUNÇÕES DE UTILIDADE (A LÓGICA)
// (Estas são as __host__ __device__ que substituem os métodos da classe)
// ===================================================================

// --- Funções Vec2f ---

__host__ __device__ inline Vec2f vec2f_add(Vec2f a, Vec2f b) {
    return {a.x + b.x, a.y + b.y};
}

__host__ __device__ inline Vec2f vec2f_sub(Vec2f a, Vec2f b) {
    return {a.x - b.x, a.y - b.y};
}

__host__ __device__ inline Vec2f vec2f_mul_scalar(Vec2f v, float s) {
    return {v.x * s, v.y * s};
}

__host__ __device__ inline Vec2f vec2f_div_scalar(Vec2f v, float s) {
    return {v.x / s, v.y / s};
}

__host__ __device__ inline float vec2f_dot(Vec2f a, Vec2f b) {
    return a.x * b.x + a.y * b.y;
}

__host__ __device__ inline float vec2f_magnitude(Vec2f v) {
    return sqrtf(vec2f_dot(v, v));
}

__host__ __device__ inline Vec2f vec2f_normalize(Vec2f v) {
    float mag = vec2f_magnitude(v);
    if (mag > 1e-6f) { // Evita divisão por zero
        return vec2f_div_scalar(v, mag);
    }
    return {0.0f, 0.0f};
}


// --- Funções Vec3f ---

__host__ __device__ inline Vec3f vec3f_add(Vec3f a, Vec3f b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ inline Vec3f vec3f_sub(Vec3f a, Vec3f b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ inline Vec3f vec3f_mul_scalar(Vec3f v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

__host__ __device__ inline Vec3f vec3f_div_scalar(Vec3f v, float s) {
    return {v.x / s, v.y / s, v.z / s};
}

__host__ __device__ inline float vec3f_dot(Vec3f a, Vec3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vec3f vec3f_cross(Vec3f a, Vec3f b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__ inline float vec3f_magnitude(Vec3f v) {
    return sqrtf(vec3f_dot(v, v));
}

__host__ __device__ inline Vec3f vec3f_normalize(Vec3f v) {
    float mag = vec3f_magnitude(v);
    // Usar fmaxf para garantir que não dividamos por zero
    // (ou um número muito pequeno) de forma segura na GPU.
    float inv_mag = 1.0f / fmaxf(mag, 1e-9f);
    if (mag <= 1e-9f) {
        return {0.0f, 0.0f, 0.0f};
    }
    return vec3f_mul_scalar(v, inv_mag);
}

__host__ __device__ inline Vec3f vec3f_reflect(Vec3f v_in, Vec3f normal) {
    // v_out = v_in - 2 * dot(v_in, normal) * normal
    float dot_vn = vec3f_dot(v_in, normal);
    Vec3f scaled_normal = vec3f_mul_scalar(normal, 2.0f * dot_vn);
    return vec3f_sub(v_in, scaled_normal);
}

// --- Funções Quaternion ---

__host__ __device__ inline Quaternion quat_conjugate(Quaternion q) {
    return {q.w, -q.x, -q.y, -q.z};
}

__host__ __device__ inline Quaternion quat_mul(Quaternion a, Quaternion b) {
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, // w
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, // x
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, // y
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w  // z
    };
}

__host__ __device__ inline Vec3f quat_rotate_vector(Quaternion q, Vec3f v) {
    // q_v = (0, v.x, v.y, v.z)
    Quaternion q_v = {0.0f, v.x, v.y, v.z};

    // q_rotated = q * q_v * q_conjugate
    Quaternion q_conj = quat_conjugate(q);
    Quaternion q_rotated = quat_mul(quat_mul(q, q_v), q_conj);

    return {q_rotated.x, q_rotated.y, q_rotated.z};
}


// --- Funções Ray ---

__host__ __device__ inline Vec3f ray_point_at(Ray r, float t) {
    // p(t) = origin + t * direction
    return vec3f_add(r.origin, vec3f_mul_scalar(r.direction, t));
}

// ===================================================================
// 3. STRUCTS DE DADOS PARA O KERNEL
// ===================================================================

typedef struct {
    unsigned char* data_ptr; // Ponteiro para os dados da textura na VRAM
    unsigned int width;
    unsigned int height;
} Texture;

typedef struct {
    unsigned int texture_index;
    Vec3f diffuse_color;
    float specular;
    float shininess;
    Vec3f emissive_color;
} Material;

typedef struct {
    float radius;
    Material material;
    Vec3f position;
    Quaternion rotation;
    Vec3f scale; // Não usado na interseção de esfera, mas mantido para consistência
} Sphere;

typedef struct {
    float t;
    Vec3f p;
    Vec3f normal;
    Material material;
} HitRecord;


// ===================================================================
// 4. FUNÇÕES DE RENDERIZAÇÃO
// ===================================================================

__device__ Vec3f texture_sample(const Texture* tex, Vec2f uv) {
    // Garante que as coordenadas UV estejam no intervalo [0, 1]
    uv.x = fmodf(uv.x, 1.0f);
    uv.y = fmodf(uv.y, 1.0f);
    if (uv.x < 0.0f) uv.x += 1.0f;
    if (uv.y < 0.0f) uv.y += 1.0f;

    int i = (int)(uv.x * tex->width);
    int j = (int)((1.0f - uv.y) * tex->height); // Pygame e muitas libs de imagem têm (0,0) no topo-esquerdo

    // Clamping para evitar acesso fora dos limites
    i = max(0, min((int)tex->width - 1, i));
    j = max(0, min((int)tex->height - 1, j));

    // A imagem é RGBA, 4 bytes por pixel
    unsigned char* pixel = tex->data_ptr + (j * tex->width + i) * 4;
    float r = pixel[0] / 255.0f;
    float g = pixel[1] / 255.0f;
    float b = pixel[2] / 255.0f;

    return {r, g, b};
}

__device__ bool intersect_sphere(const Ray* r, const Sphere* s, float t_min, float t_max, HitRecord* rec) {
    Vec3f oc = vec3f_sub(r->origin, s->position);
    float a = vec3f_dot(r->direction, r->direction);
    float half_b = vec3f_dot(oc, r->direction);
    float c = vec3f_dot(oc, oc) - s->radius * s->radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0) {
        return false;
    }

    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec->t = root;
    rec->p = ray_point_at(*r, rec->t);
    rec->normal = vec3f_normalize(vec3f_div_scalar(vec3f_sub(rec->p, s->position), s->radius));
    rec->material = s->material;

    return true;
}

__device__ bool trace(const Ray* r, const Sphere* spheres, int num_spheres, float t_min, float t_max, HitRecord* rec, int* hit_sphere_index) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    *hit_sphere_index = -1;

    for (int i = 0; i < num_spheres; ++i) {
        if (intersect_sphere(r, &spheres[i], t_min, closest_so_far, &temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            *rec = temp_rec;
            *hit_sphere_index = i;
        }
    }
    return hit_anything;
}

__device__ Vec3f per_pixel(int x, int y, int image_width, int image_height, Vec3f camera_center, Vec3f pixel00_loc, Vec3f pixel_delta_u, Vec3f pixel_delta_v, const Sphere* spheres, int num_spheres, const Texture* textures, int sky_box_index, Vec3f light_dir, Vec3f ambient_light) {
    // Calcula a direção do raio para o pixel atual
    Vec3f pixel_center = vec3f_add(pixel00_loc, vec3f_add(vec3f_mul_scalar(pixel_delta_u, x), vec3f_mul_scalar(pixel_delta_v, y)));
    Vec3f ray_direction = vec3f_normalize(vec3f_sub(pixel_center, camera_center));
    Ray r = {camera_center, ray_direction};

    HitRecord rec;
    int hit_sphere_index = -1;
    if (trace(&r, spheres, num_spheres, 0.001f, 1e10f, &rec, &hit_sphere_index)) {
        Vec3f diffuse_color = rec.material.diffuse_color;
        if (rec.material.texture_index != -1 && hit_sphere_index != -1) {
            // Calcular coordenadas UV para a esfera
            const Sphere* hit_sphere = &spheres[hit_sphere_index];
            Vec3f p_local = vec3f_sub(rec.p, hit_sphere->position);
            float phi = atan2f(p_local.z, p_local.x);
            float theta_arg = p_local.y / hit_sphere->radius;
            float theta = asinf(fmaxf(-1.0f, fminf(1.0f, theta_arg))); // Clamp para evitar NaN
            float u = 1.0f - (phi + 3.14159265f) / (2.0f * 3.14159265f);
            float v = (theta + 3.14159265f / 2.0f) / 3.14159265f;
            diffuse_color = texture_sample(&textures[rec.material.texture_index], {u, v});
        }

        // Iluminação Ambiente
        Vec3f color = vec3f_mul_scalar(diffuse_color, ambient_light.x); // Simplificado: usando x como intensidade

        // Iluminação Difusa (Lambert)
        float diff = fmaxf(vec3f_dot(rec.normal, light_dir), 0.0f);
        Vec3f diffuse = vec3f_mul_scalar(diffuse_color, diff);
        color = vec3f_add(color, diffuse);

        // Iluminação Especular (Blinn-Phong)
        Vec3f view_dir = vec3f_normalize(vec3f_sub(camera_center, rec.p));
        Vec3f reflect_dir = vec3f_reflect(vec3f_mul_scalar(light_dir, -1.0f), rec.normal);
        float spec = powf(fmaxf(vec3f_dot(view_dir, reflect_dir), 0.0f), rec.material.shininess);
        Vec3f specular = vec3f_mul_scalar({1.0f, 1.0f, 1.0f}, rec.material.specular * spec);
        color = vec3f_add(color, specular);

        // Cor Emissiva
        color = vec3f_add(color, rec.material.emissive_color);

        return color;
    }

    // Cor de fundo (Skybox)
    if (sky_box_index != -1) {
        float u = 0.5f + atan2f(ray_direction.z, ray_direction.x) / (2.0f * 3.14159265f);
        float v = 0.5f - asinf(ray_direction.y) / 3.14159265f;
        return texture_sample(&textures[sky_box_index], {u, v});
    }

    // Cor de fundo padrão se não houver skybox
    float t = 0.5f * (ray_direction.y + 1.0f);
    Vec3f start_color = {1.0f, 1.0f, 1.0f};
    Vec3f end_color = {0.5f, 0.7f, 1.0f};
    return vec3f_add(vec3f_mul_scalar(start_color, 1.0f - t), vec3f_mul_scalar(end_color, t));
}

// ===================================================================
// 5. KERNEL PRINCIPAL
// ===================================================================

extern "C" __global__ void kernel(
    unsigned char* output_image,
    int image_width,
    int image_height,
    Vec3f camera_center,
    Vec3f pixel00_loc,
    Vec3f pixel_delta_u,
    Vec3f pixel_delta_v,
    Sphere* spheres,
    int num_spheres,
    Texture* textures,
    int sky_box_index,
    Vec3f light_direction,
    Vec3f ambient_light
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= image_width || j >= image_height) {
        return;
    }

    Vec3f color = per_pixel(i, j, image_width, image_height, camera_center, pixel00_loc, pixel_delta_u, pixel_delta_v,
                            spheres, num_spheres, textures, sky_box_index, light_direction, ambient_light);

    // Clamping da cor para o intervalo [0, 1]
    color.x = fmaxf(0.0f, fminf(1.0f, color.x));
    color.y = fmaxf(0.0f, fminf(1.0f, color.y));
    color.z = fmaxf(0.0f, fminf(1.0f, color.z));

    int pixel_index = (j * image_width + i) * 3;
     
    output_image[pixel_index + 0] = (unsigned char)(255.999 * color.x);
    output_image[pixel_index + 1] = (unsigned char)(255.999 * color.y);
    output_image[pixel_index + 2] = (unsigned char)(255.999 * color.z);

//     output_image[pixel_index + 0] = 255;
//     output_image[pixel_index + 1] = 0;
//     output_image[pixel_index + 2] = 0;
}
