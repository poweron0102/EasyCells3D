#ifndef VEC_MATH_CUH
#define VEC_MATH_CUH

// Definição de M_PI se não estiver disponível
#ifndef M_PI
#define M_PI (3.14159265358979323846f)
#endif

#include <math.h>

// ===================================================================
// 1. DEFINIÇÕES DAS STRUCTS
// ===================================================================

struct Vec2f {
    float x, y;

    __host__ __device__ inline Vec2f() : x(0), y(0) {}
    __host__ __device__ inline Vec2f(float x, float y) : x(x), y(y) {}

    __host__ __device__ inline Vec2f operator+(Vec2f r) const { return Vec2f(x + r.x, y + r.y); }
    __host__ __device__ inline Vec2f operator-(Vec2f r) const { return Vec2f(x - r.x, y - r.y); }
    __host__ __device__ inline Vec2f operator*(float s) const { return Vec2f(x * s, y * s); }
    __host__ __device__ inline Vec2f operator/(float s) const { return Vec2f(x / s, y / s); }
};

struct Vec3f {
    float x, y, z;

    __host__ __device__ inline Vec3f() : x(0), y(0), z(0) {}
    __host__ __device__ inline Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ inline Vec3f operator+(Vec3f r) const { return Vec3f(x + r.x, y + r.y, z + r.z); }
    __host__ __device__ inline Vec3f operator-(Vec3f r) const { return Vec3f(x - r.x, y - r.y, z - r.z); }
    __host__ __device__ inline Vec3f operator*(float s) const { return Vec3f(x * s, y * s, z * s); }
    __host__ __device__ inline Vec3f operator/(float s) const { return Vec3f(x / s, y / s, z / s); }
    __host__ __device__ inline Vec3f operator-() const { return Vec3f(-x, -y, -z); }
};

typedef struct {
    float w;
    float x;
    float y;
    float z;
} Quaternion;

// ===================================================================
// 2. FUNÇÕES DE UTILIDADE
// ===================================================================

// --- Funções Livres (Vec2f) ---

__host__ __device__ inline float vec2f_dot(Vec2f a, Vec2f b) {
    return a.x * b.x + a.y * b.y;
}

__host__ __device__ inline float vec2f_magnitude(Vec2f v) {
    return sqrtf(vec2f_dot(v, v));
}

__host__ __device__ inline Vec2f vec2f_normalize(Vec2f v) {
    float mag = vec2f_magnitude(v);
    if (mag > 1e-6f) { // Evita divisão por zero
        return v / mag;
    }
    return Vec2f(0.0f, 0.0f);
}


// --- Funções Livres (Vec3f) ---

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
        return Vec3f(0.0f, 0.0f, 0.0f);
    }
    return v * inv_mag;
}

__host__ __device__ inline Vec3f vec3f_reflect(Vec3f v_in, Vec3f normal) {
    // v_out = v_in - 2 * dot(v_in, normal) * normal
    float dot_vn = vec3f_dot(v_in, normal);
    return v_in - normal * (2.0f * dot_vn);
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


#endif // VEC_MATH_CUH
