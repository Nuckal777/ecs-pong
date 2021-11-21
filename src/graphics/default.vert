#version 330

layout(location = 0) in vec2 vertex;
layout(location = 1) in vec3 color_in;

out vec3 frag_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main() {
    frag_color = color_in;
    mat4 mvp = proj * view * model;
    gl_Position = mvp * vec4(vertex, -10.0, 1.0);
}
