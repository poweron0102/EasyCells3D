// #include <cuda_runtime.h>
// #include <math.h>
//
// #ifndef M_PI
// #define M_PI 3.14159265358979323846
// #endif
//
// // Define vector types for clarity
// typedef float3 vec3;
// typedef float4 quat;
//
// // Define structs for scene objects
// struct Sphere {
//     vec3 center;
//     float radius;
//     int material_index;
//     quat rotation; // w, x, y, z
// };
//
// struct Material {
//     vec3 diffuse_color;
//     float specular;
//     float shininess;
//     int texture_index;
//     vec3 emissive_color;
// };
//
// // Device function for dot product
// __device__ float vec3_dot(vec3 a, vec3 b) {
//     return a.x * b.x + a.y * b.y + a.z * b.z;
// }
//
// // Device function for vector length squared
// __device__ float vec3_length_squared(vec3 a) {
//     return a.x * a.x + a.y * a.y + a.z * a.z;
// }
//
// // Device function for vector normalization
// __device__ vec3 vec3_normalize(vec3 a) {
//     float len_sq = vec3_length_squared(a);
//     if (len_sq > 0) {
//         float inv_len = rsqrtf(len_sq);
//         return make_float3(a.x * inv_len, a.y * inv_len, a.z * inv_len);
//     }
//     return a;
// }
//
// // Device function for cross product
// __device__ vec3 cross(vec3 a, vec3 b) {
//     return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
// }
//
// // Overload operators for vec3
// __device__ vec3 operator+(vec3 a, vec3 b) {
//     return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
// }
//
// __device__ vec3 operator-(vec3 a, vec3 b) {
//     return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
// }
//
// __device__ vec3 operator-(vec3 a) {
//     return make_float3(-a.x, -a.y, -a.z);
// }
//
// __device__ vec3 operator*(vec3 a, float s) {
//     return make_float3(a.x * s, a.y * s, a.z * s);
// }
//
// __device__ vec3 operator*(float s, vec3 a) {
//     return make_float3(a.x * s, a.y * s, a.z * s);
// }
//
// __device__ vec3 operator*(vec3 a, vec3 b) {
//     return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
// }
//
// // Device function for quaternion inverse
// __device__ quat quat_inverse(quat q) {
//     return make_float4(q.w, -q.x, -q.y, -q.z);
// }
//
// // Device function to rotate a vector by a quaternion
// __device__ vec3 quat_rotate_vector(quat q, vec3 v) {
//     vec3 t = 2.0f * cross(make_float3(q.x, q.y, q.z), v);
//     return v + q.w * t + cross(make_float3(q.x, q.y, q.z), t);
// }
//
//
// // Device function to get UV coordinates on a sphere
// __device__ void get_sphere_uv_gpu(vec3 world_normal, quat sphere_rotation, float& u, float& v) {
//     vec3 local_normal = quat_rotate_vector(quat_inverse(sphere_rotation), world_normal);
//     float theta = asinf(local_normal.y);
//     float phi = atan2f(local_normal.z, local_normal.x);
//     u = 1.0f - (phi + M_PI) / (2.0f * M_PI);
//     v = (theta + M_PI / 2.0f) / M_PI;
// }
//
// // Device function for ray-sphere intersection
// __device__ float sphere_intersect(vec3 ray_origin, vec3 ray_direction, const Sphere& sphere, vec3& hit_normal) {
//     vec3 oc = ray_origin - sphere.center;
//     float a = vec3_dot(ray_direction, ray_direction);
//     float half_b = vec3_dot(oc, ray_direction);
//     float c = vec3_dot(oc, oc) - sphere.radius * sphere.radius;
//     float discriminant = half_b * half_b - a * c;
//
//     if (discriminant < 0) {
//         return -1.0f;
//     }
//
//     float sqrt_d = sqrtf(discriminant);
//     float root = (-half_b - sqrt_d) / a;
//     if (root <= 0.001f) {
//         root = (-half_b + sqrt_d) / a;
//         if (root <= 0.001f) {
//             return -1.0f;
//         }
//     }
//
//     vec3 hit_point = ray_origin + root * ray_direction;
//     hit_normal = vec3_normalize(hit_point - sphere.center);
//     return root;
// }
//
// // Device function for texture lookup
// __device__ vec3 texture_lookup(const unsigned char* textures_data, const int3* texture_info, int texture_index, float u, float v) {
//     v = 1.0f - v; // Invert v
//
//     int3 info = texture_info[texture_index];
//     int offset = info.x;
//     int width = info.y;
//     int height = info.z;
//
//     int x = static_cast<int>(u * (width - 1));
//     int y = static_cast<int>(v * (height - 1));
//
//     x = max(0, min(width - 1, x));
//     y = max(0, min(height - 1, y));
//
//     int pixel_offset = offset + (y * width + x) * 3; // 3 channels (RGB)
//     float r = textures_data[pixel_offset + 0] / 255.0f;
//     float g = textures_data[pixel_offset + 1] / 255.0f;
//     float b = textures_data[pixel_offset + 2] / 255.0f;
//
//     return make_float3(r, g, b);
// }
//
//
// // Main CUDA kernel
// extern "C" __global__ void render_kernel(
//     unsigned char* pixel_array,
//     int width, int height,
//     vec3 camera_center,
//     vec3 pixel00_loc,
//     vec3 pixel_delta_u,
//     vec3 pixel_delta_v,
//     const Sphere* spheres,
//     int num_spheres,
//     const Material* materials,
//     int num_materials,
//     const unsigned char* textures_data,
//     const int3* texture_info,
//     vec3 light_direction,
//     vec3 ambient_light
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//     if (i >= width || j >= height) {
//         return;
//     }
//
//     // 1. Calculate ray
//     vec3 pixel_center = pixel00_loc + (float)i * pixel_delta_u + (float)j * pixel_delta_v;
//     vec3 ray_direction = vec3_normalize(pixel_center - camera_center);
//
//     // 2. Ray-scene intersection
//     float min_dist = 1e10f;
//     vec3 hit_normal = make_float3(0.0f, 0.0f, 0.0f);
//     int hit_material_index = -1;
//     quat hit_sphere_rotation = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
//     bool hit = false;
//
//     for (int s = 0; s < num_spheres; ++s) {
//         vec3 current_normal;
//         float dist = sphere_intersect(camera_center, ray_direction, spheres[s], current_normal);
//
//         if (dist > 0.001f && dist < min_dist) {
//             min_dist = dist;
//             hit_normal = current_normal;
//             hit_material_index = spheres[s].material_index;
//             hit_sphere_rotation = spheres[s].rotation;
//             hit = true;
//         }
//     }
//
//     // 3. Shading
//     vec3 final_color;
//     if (hit) {
//         const Material& mat = materials[hit_material_index];
//         vec3 albedo;
//
//         if (mat.texture_index != -1) {
//             float u, v;
//             get_sphere_uv_gpu(hit_normal, hit_sphere_rotation, u, v);
//             albedo = texture_lookup(textures_data, texture_info, mat.texture_index, u, v);
//         } else {
//             albedo = mat.diffuse_color;
//         }
//
//         vec3 emissive = mat.emissive_color;
//         vec3 ambient = albedo * ambient_light;
//
//         float diffuse_intensity = max(0.0f, vec3_dot(hit_normal, light_direction));
//         vec3 diffuse = albedo * diffuse_intensity;
//
//         vec3 view_dir = vec3_normalize(-ray_direction);
//         vec3 half_vector = vec3_normalize(light_direction + view_dir);
//         float specular_intensity = powf(max(0.0f, vec3_dot(hit_normal, half_vector)), mat.shininess);
//         vec3 specular = make_float3(mat.specular, mat.specular, mat.specular) * specular_intensity;
//
//         final_color = emissive + ambient + diffuse + specular;
//
//     } else {
//         // Background color
//         float a = 0.5f * (ray_direction.y + 1.0f);
//         final_color = (1.0f - a) * make_float3(1.0f, 1.0f, 1.0f) + a * make_float3(0.5f, 0.7f, 1.0f);
//     }
//
//     // 4. Write to pixel array
//     int pixel_index = (j * width + i) * 4; // RGBA
//     pixel_array[pixel_index + 0] = static_cast<unsigned char>(255.999f * fminf(1.0f, final_color.x));
//     pixel_array[pixel_index + 1] = static_cast<unsigned char>(255.999f * fminf(1.0f, final_color.y));
//     pixel_array[pixel_index + 2] = static_cast<unsigned char>(255.999f * fminf(1.0f, final_color.z));
//     pixel_array[pixel_index + 3] = 255;
// }