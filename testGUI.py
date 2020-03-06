import numpy as np
from glumpy import app, gl, gloo, data, log, glm
from glumpy.transforms import Trackball, Position
import random,os,sys
import matplotlib.pyplot as plt
import cv2
from utils import player,soundmodel

def load_shader(path):
    with open(path + '.vert', 'r') as f:
        vertex = f.read()
    with open(path + '.frag', 'r') as f:
        fragment = f.read()
    return gloo.Program(vertex, fragment)

class GUI(object):
	def __init__(self, filename):
		self.mesh_shader = load_shader('shader/mesh')
		trackball = Trackball(Position("position"))
		self.mesh_shader['transform'] = trackball
		trackball.theta, trackball.phi, trackball.zoom = 80, -135, 5

		self.img_shader = load_shader('shader/image')
		self.img_shader['position']= [(-1,-1), (-1,1), (1,-1), (1,1)]

		self.window = self.set_window()
		self.mouse = None
		self.player = player.PlayerWorld()


		self.load_mesh(filename)
		app.run()

	def click(self, index):
		a,w,c = self.model.click(index,100)
		self.player.play(a,w,c)

	def load_mesh(self,filename):
		mat = 4
		self.model = soundmodel.mesh(filename, mat)
		vertices = self.model.vertices
		faces = self.model.faces
		normals = self.model.noramls

		V = np.zeros(len(faces)*3, [("position", np.float32, 3),
										("normal", np.float32, 3),
										("id", np.float32, 1)])
		V['position'] = vertices[faces].reshape(-1,3)
		V['normal'] = normals[faces].reshape(-1,3)
		V['id'] = (np.arange(0,len(faces)*3)//3 + 1)
		V = V.view(gloo.VertexBuffer)
		self.mesh_shader.bind(V)
		self.mesh_shader['select_id'] = -1

	def render(self):
		self.img_shader.draw(gl.GL_TRIANGLE_STRIP)

	def update(self):
		model = self.mesh_shader['transform']['model'].reshape(4,4)
		view  = self.mesh_shader['transform']['view'].reshape(4,4)
		self.mesh_shader['m_view']  = view
		self.mesh_shader['m_model'] = model
		self.mesh_shader['m_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)

	def set_framebuffer(self, width, height):
		color = np.zeros((height,width,4),np.ubyte).view(gloo.Texture2D)
		color.interpolation = gl.GL_LINEAR
		pick = np.zeros((height,width,4),np.ubyte).view(gloo.Texture2D)
		pick.interpolation = gl.GL_LINEAR
		framebuffer = gloo.FrameBuffer(color=[color,pick], depth=gloo.DepthBuffer(width, height))
		self.framebuffer = framebuffer
		self.img_shader["color"] = self.framebuffer.color[0]

	def set_window(self):
		window = app.Window(width=1024, height=768)
		self.set_framebuffer(window.width,window.height)
		@window.event
		def on_draw(dt):
			gl.glEnable(gl.GL_DEPTH_TEST)
			self.framebuffer.activate()
			window.clear()
			self.mesh_shader.draw(gl.GL_TRIANGLES)
			if self.mouse is not None:
				gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
				r,g,b,a = gl.glReadPixels(self.mouse[0],self.mouse[1],1,1, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
				if type(r) is not int: r = ord(r)
				if type(g) is not int: g = ord(g)
				if type(b) is not int: b = ord(b)
				index = b + 256*g + 256*256*r
				self.mesh_shader['select_id'] = index
				self.mouse = None
				self.click(index - 1)
			self.framebuffer.deactivate()
			window.clear()
			self.render()
			
			
		@window.event
		def on_mouse_drag(x, y, dx, dy, button):
			self.update()
			
		@window.event
		def on_init():
			gl.glEnable(gl.GL_DEPTH_TEST)
			self.update()

		@window.event
		def on_resize(width, height):
			self.set_framebuffer(width, height)

		@window.event
		def on_mouse_press(x, y, button):
			if (button == 8): #right click
				self.mouse = int(x), window.height-int(y)
		
		@window.event
		def on_key_press(symbol, modifiers):
			#print(symbol)
			if symbol >= 48 and symbol < 58:
				self.model.change_mat(symbol - 48) 
			if symbol == 65363:
				self.load_mesh(1)
			if symbol == 65361:
				self.load_mesh(-1)
			if symbol == 65362:
				self.model.rescale(1.1)
			if symbol == 65364:
				self.model.rescale(0.9)
		window.attach(self.mesh_shader['transform'])
		return window
        

if __name__ == "__main__":
	name1 = 'originData\\datasetGreat\\bowl_0005.off.ply.ply'
	name2 = 'originData\\dataset\\train\\bowl_0005.off.ply'
	GUI(name2)
