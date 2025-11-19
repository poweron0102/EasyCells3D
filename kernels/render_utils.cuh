#ifndef RENDER_UTILS_CUH
#define RENDER_UTILS_CUH

#include "ray_tracing.cuh"

// ===================================================================
// FUNÇÕES DE RENDERIZAÇÃO
// ===================================================================

__device__ Vec3f texture_sample(const Texture* tex, Vec2f uv) {
    // Garante que as coordenadas UV estejam no intervalo [0, 1]
    uv.x = fmodf(uv.x, 1.0f);
    uv.y = fmodf(uv.y, 1.0f);
    if (uv.x < 0.0f) uv.x += 1.0f;
    if (uv.y < 0.0f) uv.y += 1.0f;

    // Mapeamento UV padrão
    int i = (int)(uv.x * tex->width);
    int j = (int)((1.0f - uv.y) * tex->height);

    // Clamping
    i = max(0, min((int)tex->width - 1, i));
    j = max(0, min((int)tex->height - 1, j));

    int pixel_index = (i * tex->height + j);

    int channels = 3; // Mude para 4 se estiver usando pixels_alpha ou imagens convertidas para RGBA
    unsigned char* pixel = tex->data_ptr + pixel_index * channels;

    float r = pixel[0] / 255.0f;
    float g = pixel[1] / 255.0f;
    float b = pixel[2] / 255.0f;

    return {r, g, b};
}


__device__ Vec3f per_pixel(int x, int y, int image_width, int image_height, Vec3f camera_center, Vec3f pixel00_loc, Vec3f pixel_delta_u, Vec3f pixel_delta_v, const Sphere* spheres, int num_spheres, const Texture* textures, int sky_box_index, Vec3f light_dir, Vec3f ambient_light) {
    // Calcula a direção do raio para o pixel atual
    Vec3f pixel_center = pixel00_loc + (pixel_delta_u * x) + (pixel_delta_v * y);
    Vec3f ray_direction = vec3f_normalize(pixel_center - camera_center);
    Ray r = {camera_center, ray_direction};

    TraceResult trace_res = trace(&r, spheres, num_spheres, 0.001f, 1e10f);
    if (trace_res.hit) {
        Vec3f diffuse_color = trace_res.material.diffuse_color;
        if (trace_res.material.texture_index != -1) {
            diffuse_color = texture_sample(&textures[trace_res.material.texture_index], trace_res.uv);
        }

        // Iluminação Ambiente
        Vec3f color = diffuse_color * ambient_light.x; // Simplificado: usando x como intensidade

        // Iluminação Difusa (Lambert)
        float diff = fmaxf(vec3f_dot(trace_res.rec.normal, light_dir), 0.0f);
        Vec3f diffuse = diffuse_color * diff;
        color = color + diffuse;

        // Iluminação Especular (Blinn-Phong)
        Vec3f view_dir = vec3f_normalize(camera_center - trace_res.rec.p);
        Vec3f reflect_dir = vec3f_reflect(-light_dir, trace_res.rec.normal);
        float spec = powf(fmaxf(vec3f_dot(view_dir, reflect_dir), 0.0f), trace_res.material.shininess);
        Vec3f specular = Vec3f(1.0f, 1.0f, 1.0f) * (trace_res.material.specular * spec);
        color = color + specular;

        // Cor Emissiva
        color = color + trace_res.material.emissive_color;

        return color;
    }

    // Cor de fundo (Skybox)
    if (sky_box_index != -1) {
        float u = 0.5f + atan2f(ray_direction.z, ray_direction.x) / (2.0f * M_PI);
        float v = 0.5f - asinf(ray_direction.y) / M_PI;
        return texture_sample(&textures[sky_box_index], Vec2f(u, v));
    }

    // Cor de fundo padrão se não houver skybox
    float t = 0.5f * (ray_direction.y + 1.0f);
    Vec3f start_color = {1.0f, 1.0f, 1.0f};
    Vec3f end_color = {0.5f, 0.7f, 1.0f};
    return start_color * (1.0f - t) + end_color * t;
}


#endif // RENDER_UTILS_CUH
