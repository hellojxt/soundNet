#include "math/constants.glsl"
uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_normal;
attribute vec3 position;
attribute vec3 normal;
attribute float id;
varying vec3 v_normal;
varying vec3 v_position;
varying vec4 v_id;
varying float origin_id;
void main()
{
    v_id = vec4 ( mod(floor(id / (256*256)), 256) / 255.0,
                  mod(floor(id /     (256)), 256) / 255.0,
                  mod(floor(id /       (1)), 256) / 255.0,
                  1.0 );
    origin_id = id;
    gl_Position = <transform>;
    vec4 P = m_view * m_model* vec4(position, 1.0);
    v_position = P.xyz / P.w;
    v_normal = vec3(m_normal * vec4(normal,0.0));
}