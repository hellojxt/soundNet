attribute vec2 position;
varying vec2 v_texcoord;
void main(void)
{
    gl_Position = vec4(position,0,1);
    v_texcoord = (position+1.0)/2.0;
}