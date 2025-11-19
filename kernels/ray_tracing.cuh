#ifndef RAY_TRACING_CUH
#define RAY_TRACING_CUH

#include "vec_math.cuh"

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
// 2. FUNÇÕES DE RAY TRACING
// ===================================================================

__host__ __device__ inline Vec3f ray_point_at(Ray r, float t) {
    // p(t) = origin + t * direction
    return r.origin + r.direction * t;
}

__device__ bool intersect_sphere(const Ray* r, const Sphere* s, float t_min, float t_max, HitRecord* rec) {
    Vec3f oc = r->origin - s->position;
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
    rec->normal = vec3f_normalize((rec->p - s->position) / s->radius);
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


#endif // RAY_TRACING_CUH
