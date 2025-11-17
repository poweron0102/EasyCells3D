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




