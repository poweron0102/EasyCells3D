#ifndef RAY_TRACING_CUH
#define RAY_TRACING_CUH

#include "vec_math.cuh"
#include <stdio.h>

// ===================================================================
// 1. STRUCTS DE DADOS PARA O KERNEL
// ===================================================================

typedef struct {
    Vec3f origin;
    Vec3f direction;
} Ray;

typedef struct {
    unsigned char* data_ptr; // Ponteiro para os dados da textura na VRAM
    unsigned int width;
    unsigned int height;
} Texture;

typedef struct {
    int texture_index;
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
    Vec3f scale;
} Sphere;

typedef struct {
    // 1. Ponteiros (8 bytes cada -> Alinhamento 8)
    int* voxels_ptr;              // Offset 0
    Material* materials_ptr;      // Offset 8

    // 2. Dados geométricos e contadores (4 bytes cada componente -> Alinhamento 4)
    Vec3f position;               // Offset 16 (12 bytes) -> Fim 28
    Quaternion rotation;          // Offset 28 (16 bytes) -> Fim 44
    Vec3f scale;                  // Offset 44 (12 bytes) -> Fim 56
    unsigned int voxels_size[3];  // Offset 56 (12 bytes) -> Fim 68
    unsigned int num_materials;   // Offset 68 (4 bytes)  -> Fim 72
} Voxels; 
// Tamanho Total: 72 bytes (Múltiplo de 8, então não gera padding final)

typedef struct {
    float t;
    Vec3f p;
    Vec3f normal;
    Material material;
} HitRecord;

typedef struct {
    bool hit;
    Material material;
    Vec2f uv;
    HitRecord rec;
} TraceResult;


// ===================================================================
// 2. FUNÇÕES DE RAY TRACING
// ===================================================================

__host__ __device__ inline Vec3f ray_point_at(Ray r, float t) {
    return r.origin + r.direction * t;
}

__device__ bool intersect_sphere(const Ray* r, const Sphere* s, float t_min, float t_max, HitRecord* rec) {
    // Transformar o raio para o espaço local da esfera (considerando posição, rotação e escala)
    Vec3f oc = r->origin - s->position;

    Quaternion inv_rot = quat_conjugate(s->rotation);
    Vec3f ray_origin_local = quat_rotate_vector(inv_rot, oc);
    Vec3f ray_dir_local = quat_rotate_vector(inv_rot, r->direction);

    // Aplicar escala inversa
    Vec3f inv_scale = {1.0f / s->scale.x, 1.0f / s->scale.y, 1.0f / s->scale.z};
    ray_origin_local = vec3f_mul_comp(ray_origin_local, inv_scale);
    ray_dir_local = vec3f_mul_comp(ray_dir_local, inv_scale);

    // Agora, fazemos a interseção com uma esfera de raio `s->radius` na origem.
    float a = vec3f_dot(ray_dir_local, ray_dir_local);
    float half_b = vec3f_dot(ray_origin_local, ray_dir_local);
    float c = vec3f_dot(ray_origin_local, ray_origin_local) - s->radius * s->radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0) {
        return false;
    }

    float sqrt_d = sqrtf(discriminant);
    float root = (-half_b - sqrt_d) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrt_d) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec->t = root;
    rec->p = ray_point_at(*r, rec->t);

    // Transformar a normal do espaço local para o espaço global
    Vec3f p_local = ray_origin_local + ray_dir_local * root;
    Vec3f normal_local = vec3f_normalize(p_local / s->radius);
    rec->normal = vec3f_normalize(quat_rotate_vector(s->rotation, vec3f_mul_comp(normal_local, inv_scale)));
    rec->material = s->material;

    return true;
}

__device__ Vec2f get_sphere_uv(const Vec3f* p_world, const Sphere* sphere) {
    // Transforma o ponto de colisão do espaço do mundo para o espaço local da esfera
    // para calcular as coordenadas UV corretamente, considerando rotação e escala.
    Vec3f p_relative_world = *p_world - sphere->position;

    Quaternion inv_rot = quat_conjugate(sphere->rotation);
    Vec3f p_rotated = quat_rotate_vector(inv_rot, p_relative_world);

    Vec3f inv_scale = {1.0f / sphere->scale.x, 1.0f / sphere->scale.y, 1.0f / sphere->scale.z};
    Vec3f p_local = vec3f_mul_comp(p_rotated, inv_scale);

    // Calcula as coordenadas esféricas (phi, theta) para mapeamento UV
    float phi = atan2f(p_local.z, p_local.x);
    float theta_arg = p_local.y / sphere->radius;
    float theta = asinf(fmaxf(-1.0f, fminf(1.0f, theta_arg))); // Clamp para segurança numérica

    float u = 1.0f - (phi + M_PI) / (2.0f * M_PI);
    float v = (theta + M_PI / 2.0f) / M_PI;
    return {u, v};
}

__device__ bool intersect_voxels(const Ray* r, const Voxels* v, float t_min, float t_max, HitRecord* rec) {
    // Passo 1: Transformar o raio para o espaço local dos voxels
    Vec3f oc = r->origin - v->position;
    Quaternion inv_rot = quat_conjugate(v->rotation);
    Vec3f ray_origin_local = quat_rotate_vector(inv_rot, oc);
    Vec3f ray_dir_local = quat_rotate_vector(inv_rot, r->direction);

    // A escala do objeto define o tamanho do volume de voxels.
    // Para o DDA, normalizamos o espaço para que a grade tenha voxels de tamanho 1x1x1.
    // A "escala do voxel" é a escala do objeto dividida pelas dimensões da grade.
    Vec3f grid_dims = { (float)v->voxels_size[0], (float)v->voxels_size[1], (float)v->voxels_size[2] };
    Vec3f inv_scale = { grid_dims.x / v->scale.x, grid_dims.y / v->scale.y, grid_dims.z / v->scale.z };

    // Centraliza a origem da grade em (0,0,0)
    Vec3f grid_center_offset = grid_dims * 0.5f;
    Vec3f grid_origin = vec3f_mul_comp(ray_origin_local, inv_scale) + grid_center_offset;
    Vec3f grid_dir = vec3f_mul_comp(ray_dir_local, inv_scale);

    // Passo 2: Interseção com o AABB da grade de voxels
    // O AABB agora vai de (0,0,0) até grid_dims, pois a origem do raio foi deslocada.
    Vec3f inv_dir = { 1.0f / grid_dir.x, 1.0f / grid_dir.y, 1.0f / grid_dir.z };
    Vec3f t1 = vec3f_mul_comp(Vec3f(0.0f, 0.0f, 0.0f) - grid_origin, inv_dir);
    Vec3f t2 = vec3f_mul_comp(grid_dims - grid_origin, inv_dir);

    float t_entry_x = fminf(t1.x, t2.x); float t_exit_x = fmaxf(t1.x, t2.x); // NOLINT
    float t_entry_y = fminf(t1.y, t2.y); float t_exit_y = fmaxf(t1.y, t2.y);
    float t_entry_z = fminf(t1.z, t2.z); float t_exit_z = fmaxf(t1.z, t2.z);

    float t_entry = fmaxf(fmaxf(t_entry_x, t_entry_y), t_entry_z);
    float t_exit = fminf(fminf(t_exit_x, t_exit_y), t_exit_z);

    if (t_entry >= t_exit || t_exit < 0) {
        return false;
    }

    // Passo 3: Algoritmo DDA 3D
    float start_dist = fmaxf(0.0f, t_entry);
    Vec3f current_pos = grid_origin + grid_dir * start_dist;

    int voxel_x = (int)floorf(current_pos.x);
    int voxel_y = (int)floorf(current_pos.y);
    int voxel_z = (int)floorf(current_pos.z);

    int step_x = (grid_dir.x > 0) ? 1 : -1;
    int step_y = (grid_dir.y > 0) ? 1 : -1;
    int step_z = (grid_dir.z > 0) ? 1 : -1;

    Vec3f t_delta = { fabsf(1.0f / grid_dir.x), fabsf(1.0f / grid_dir.y), fabsf(1.0f / grid_dir.z) };

    Vec3f t_max_dda;
    t_max_dda.x = (step_x > 0) ? (voxel_x + 1 - current_pos.x) * t_delta.x : (current_pos.x - voxel_x) * t_delta.x;
    t_max_dda.y = (step_y > 0) ? (voxel_y + 1 - current_pos.y) * t_delta.y : (current_pos.y - voxel_y) * t_delta.y;
    t_max_dda.z = (step_z > 0) ? (voxel_z + 1 - current_pos.z) * t_delta.z : (current_pos.z - voxel_z) * t_delta.z;

    Vec3f normal_local_grid = {0, 0, 0};

    while (true) {
        // Checagem de colisão
        if (voxel_x >= 0 && voxel_x < v->voxels_size[0] &&
            voxel_y >= 0 && voxel_y < v->voxels_size[1] &&
            voxel_z >= 0 && voxel_z < v->voxels_size[2])
        {
            int voxel_index = voxel_x * v->voxels_size[1] * v->voxels_size[2] + voxel_y * v->voxels_size[2] + voxel_z;
            int material_index = v->voxels_ptr[voxel_index];

            if (material_index != -1) {
                // Colisão!
                float hit_dist_grid = 0.0f;
                if (normal_local_grid.x != 0) hit_dist_grid = t_max_dda.x - t_delta.x;
                else if (normal_local_grid.y != 0) hit_dist_grid = t_max_dda.y - t_delta.y;
                else hit_dist_grid = t_max_dda.z - t_delta.z;

                float final_dist = start_dist + hit_dist_grid;

                // A distância final está no espaço de grade. Precisamos convertê-la de volta para o espaço do mundo.
                // O fator de escala é o comprimento do vetor de direção do raio no espaço da grade.
                float grid_to_world_scale = vec3f_magnitude(ray_dir_local) / vec3f_magnitude(grid_dir);
                rec->t = final_dist * grid_to_world_scale;

                if (rec->t < t_min || rec->t > t_max) return false;

                rec->p = ray_point_at(*r, rec->t);

                // Transformar a normal do espaço da grade para o espaço do mundo
                Vec3f normal_local = vec3f_normalize(vec3f_mul_comp(normal_local_grid, inv_scale));
                rec->normal = vec3f_normalize(quat_rotate_vector(v->rotation, normal_local));
                rec->material = v->materials_ptr[material_index];

                return true;
            }
        }

        // Passo do DDA
        if (t_max_dda.x < t_max_dda.y) {
            if (t_max_dda.x < t_max_dda.z) {
                if (t_max_dda.x > t_exit) return false;
                voxel_x += step_x;
                t_max_dda.x += t_delta.x;
                normal_local_grid = {- (float)step_x, 0, 0};
            } else {
                if (t_max_dda.z > t_exit) return false;
                voxel_z += step_z;
                t_max_dda.z += t_delta.z;
                normal_local_grid = {0, 0, - (float)step_z};
            }
        } else {
            if (t_max_dda.y < t_max_dda.z) {
                if (t_max_dda.y > t_exit) return false;
                voxel_y += step_y;
                t_max_dda.y += t_delta.y;
                normal_local_grid = {0, - (float)step_y, 0};
            } else {
                if (t_max_dda.z > t_exit) return false;
                voxel_z += step_z;
                t_max_dda.z += t_delta.z;
                normal_local_grid = {0, 0, - (float)step_z};
            }
        }
    }

    return false;
}

__device__ TraceResult trace(const Ray* r, const Sphere* spheres, int num_spheres, const Voxels* voxels, int num_voxels, float t_min, float t_max) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    int hit_sphere_index = -1;
    TraceResult result;
    result.hit = false;

    for (int i = 0; i < num_spheres; ++i) {
        if (intersect_sphere(r, &spheres[i], t_min, closest_so_far, &temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            result.rec = temp_rec;
            hit_sphere_index = i;
        }
    }

    for (int i = 0; i < num_voxels; ++i) {
        if (intersect_voxels(r, &voxels[i], t_min, closest_so_far, &temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            result.rec = temp_rec;
            hit_sphere_index = -1; // Indica que não foi uma esfera
        }
    }

    if (hit_anything) {
        result.hit = true;
        result.material = result.rec.material;

        if (hit_sphere_index != -1) {
            result.uv = get_sphere_uv(&result.rec.p, &spheres[hit_sphere_index]);
        } // Nota: UVs para voxels não estão implementados nesta versão.
    }
    return result;
}


#endif // RAY_TRACING_CUH
