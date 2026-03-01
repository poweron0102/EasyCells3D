#version 330

// Atributos de entrada (fornecidos automaticamente pelo Raylib)
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;

// Uniforms (fornecidos automaticamente pelo Raylib)
uniform mat4 mvp;

// Saídas para o Fragment Shader
out vec3 fragNormal;
out vec2 fragTexCoord;

void main()
{
    fragNormal = vertexNormal;
    fragTexCoord = vertexTexCoord;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}