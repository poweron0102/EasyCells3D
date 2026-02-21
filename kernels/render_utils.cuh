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


__device__ Vec3f per_pixel(int x, int y, int image_width, int image_height, Vec3f camera_center, Vec3f pixel00_loc, Vec3f pixel_delta_u, Vec3f pixel_delta_v, const Sphere* spheres, int num_spheres, const Voxels* voxels, int num_voxels, const Texture* textures, int sky_box_index, Vec3f light_dir, Vec3f ambient_light) {
    // Calcula a direção do raio para o pixel atual
    Vec3f pixel_center = pixel00_loc + (pixel_delta_u * x) + (pixel_delta_v * y);
    Vec3f ray_direction = vec3f_normalize(pixel_center - camera_center);
    Ray r = {camera_center, ray_direction};

    TraceResult trace_res = trace(&r, spheres, num_spheres, voxels, num_voxels, 0.001f, 1e10f);
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

// ===================================================================
// (Sombras + Reflexos)
// ===================================================================
__device__ Vec3f per_pixel_advanced(int x, int y, int image_width, int image_height, Vec3f camera_center, Vec3f pixel00_loc, Vec3f pixel_delta_u, Vec3f pixel_delta_v, const Sphere* spheres, int num_spheres, const Voxels* voxels, int num_voxels, const Texture* textures, int sky_box_index, Vec3f light_dir, Vec3f ambient_light, int max_bounces) {

    // Configuração inicial do raio
    Vec3f pixel_center = pixel00_loc + (pixel_delta_u * x) + (pixel_delta_v * y);
    Vec3f ray_direction = vec3f_normalize(pixel_center - camera_center);
    Ray current_ray = {camera_center, ray_direction};

    Vec3f final_color = {0.0f, 0.0f, 0.0f};
    Vec3f attenuation = {1.0f, 1.0f, 1.0f}; // Quanto de luz esse caminho ainda carrega

    for (int bounce = 0; bounce < max_bounces; bounce++) {
        TraceResult trace_res = trace(&current_ray, spheres, num_spheres, voxels, num_voxels, 0.001f, 1e10f);

        if (trace_res.hit) {
            // 1. Cor base e Textura
            Vec3f albedo = trace_res.material.diffuse_color;
            if (trace_res.material.texture_index != -1) {
                albedo = texture_sample(&textures[trace_res.material.texture_index], trace_res.uv);
            }

            // 2. Emissivo (Adiciona luz diretamente)
            final_color = final_color + vec3f_mul_comp(attenuation, trace_res.material.emissive_color);

            // 3. Sombras (Shadow Ray)
            // Lança um raio do ponto de colisão em direção à luz
            Ray shadow_ray;
            shadow_ray.origin = trace_res.rec.p + trace_res.rec.normal * 0.001f; // Bias para evitar "acne"
            shadow_ray.direction = light_dir;

            TraceResult shadow_res = trace(&shadow_ray, spheres, num_spheres, voxels, num_voxels, 0.001f, 1e10f);
            bool in_shadow = shadow_res.hit;

            // 4. Iluminação Local (Phong/Blinn)
            Vec3f local_light = {0.0f, 0.0f, 0.0f};

            // Ambiente (sempre presente)
            local_light = local_light + albedo * ambient_light.x;

            // Se não estiver na sombra, calcula difusa e especular da luz direta
            if (!in_shadow) {
                // Difusa
                float diff = fmaxf(vec3f_dot(trace_res.rec.normal, light_dir), 0.0f);
                local_light = local_light + (albedo * diff);

                // Especular (Brilho da luz)
                // Usamos o raio de visão inverso (de onde viemos)
                Vec3f view_dir = vec3f_normalize(-current_ray.direction);
                Vec3f reflect_dir = vec3f_reflect(-light_dir, trace_res.rec.normal);
                float spec_angle = fmaxf(vec3f_dot(view_dir, reflect_dir), 0.0f);
                float spec = powf(spec_angle, trace_res.material.shininess);

                Vec3f specular_color = Vec3f(1.0f, 1.0f, 1.0f) * (trace_res.material.specular * spec);
                local_light = local_light + specular_color;
            }

            // Acumula a luz local ponderada pela atenuação atual
            final_color = final_color + vec3f_mul_comp(attenuation, local_light);

            // 5. Preparar para o próximo pulo (Reflexão)
            // Se o material for especular, ele reflete o ambiente
            if (trace_res.material.specular > 0.0f) {
                // Atenua a luz para o próximo raio baseado na especularidade
                // (Materiais menos especulares refletem menos luz)
                attenuation = attenuation * trace_res.material.specular;

                // Novo raio de reflexão
                current_ray.origin = trace_res.rec.p + trace_res.rec.normal * 0.001f;
                current_ray.direction = vec3f_reflect(current_ray.direction, trace_res.rec.normal);
            } else {
                // Se não reflete, terminamos o caminho aqui
                break;
            }

        } else {
            // Miss: Atingiu o céu (Skybox)
            Vec3f sky_color;
            if (sky_box_index != -1) {
                float u = 0.5f + atan2f(current_ray.direction.z, current_ray.direction.x) / (2.0f * M_PI);
                float v = 0.5f - asinf(current_ray.direction.y) / M_PI;
                sky_color = texture_sample(&textures[sky_box_index], Vec2f(u, v));
            } else {
                float t = 0.5f * (current_ray.direction.y + 1.0f);
                Vec3f start_color = {1.0f, 1.0f, 1.0f};
                Vec3f end_color = {0.5f, 0.7f, 1.0f};
                sky_color = start_color * (1.0f - t) + end_color * t;
            }

            final_color = final_color + vec3f_mul_comp(attenuation, sky_color);
            break; // O raio escapou para o infinito
        }
    }

    return final_color;
}


#endif // RENDER_UTILS_CUH
