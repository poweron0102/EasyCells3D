#version 330

// Entradas vindas do Vertex Shader
in vec3 fragNormal;
in vec2 fragTexCoord;

// Uniforms
uniform vec4 colDiffuse; // A cor definida no construtor da Sphere (self.color)

out vec4 finalColor;

void main()
{
    // Normalização básica para garantir vetores unitários
    vec3 normal = normalize(fragNormal);

    // Iluminação simples (Direcional fixa)
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float lightDot = max(dot(normal, lightDir), 0.0);
    float ambient = 0.4;
    float lighting = ambient + lightDot;

    // Mapeia a normal (-1 a 1) para cores (0 a 1) e multiplica pela cor do objeto e luz
    vec3 normalColor = normal * 0.5 + 0.5;
    
    finalColor = vec4(colDiffuse.rgb * normalColor * lighting, colDiffuse.a);
}