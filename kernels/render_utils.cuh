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

    // --- CORREÇÃO 1: INDEXAÇÃO (Pygame Style) ---
    // Padrão C: j * width + i  (Linha por linha)
    // Pygame:   i * height + j (Coluna por coluna)
    int pixel_index = (i * tex->height + j);

    // --- CORREÇÃO 2: NÚMERO DE CANAIS ---
    // Se sua textura vem de pg.surfarray.pixels3d, ela tem 3 bytes (RGB).
    // Se o código original usava * 4, ele leria memória errada (RGBA).
    // Ajuste aqui conforme sua textura (3 para RGB, 4 para RGBA).

    int channels = 3; // Mude para 4 se estiver usando pixels_alpha ou imagens convertidas para RGBA
    unsigned char* pixel = tex->data_ptr + pixel_index * channels;

    float r = pixel[0] / 255.0f;
    float g = pixel[1] / 255.0f;
    float b = pixel[2] / 255.0f;

    // Se for RGBA, o pixel[3] seria o alpha

    return {r, g, b};
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


#endif // RENDER_UTILS_CUH
