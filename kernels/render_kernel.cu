#include "render_utils.cuh"
#include <stdio.h>

// ===================================================================
// KERNEL PRINCIPAL
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
    Voxels* voxels,
    int num_voxels,
    Texture* textures,
    int sky_box_index,
    Vec3f light_direction,
    Vec3f ambient_light
) {
    // Mapeia i para X (Largura) e j para Y (Altura)
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Coordenada X
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Coordenada Y

    // Proteção de limites
    if (i >= image_width || j >= image_height) {
        return;
    }

    int pixel_index = (i * image_height + j) * 3;

    Vec3f color = per_pixel(i, j, image_width, image_height, camera_center, pixel00_loc, pixel_delta_u, pixel_delta_v,
                            spheres, num_spheres, voxels, num_voxels, textures, sky_box_index, light_direction, ambient_light);

    // Clamping da cor para o intervalo [0, 1]
    color.x = fmaxf(0.0f, fminf(1.0f, color.x));
    color.y = fmaxf(0.0f, fminf(1.0f, color.y));
    color.z = fmaxf(0.0f, fminf(1.0f, color.z));

    // Escreve a cor no buffer de saída
    output_image[pixel_index + 0] = (unsigned char)(255.999 * color.x);
    output_image[pixel_index + 1] = (unsigned char)(255.999 * color.y);
    output_image[pixel_index + 2] = (unsigned char)(255.999 * color.z);
}
