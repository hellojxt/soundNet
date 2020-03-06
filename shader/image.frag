uniform sampler2D color;
varying vec2 v_texcoord;
void main(void)
{
    gl_FragColor = texture2D(color,v_texcoord);
}