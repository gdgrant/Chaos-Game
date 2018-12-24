from parameters import *
from multiprocessing import Pool
import numpy as np


def vertex(args):
	i, prev = args
	ids = np.arange(len(par['vertices']))
	ids = np.setdiff1d(ids, prev)
	prs = softmax(np.array(par['likelihood'])[ids])
	return np.random.choice(ids, p=prs)

class ChaosGame:

	def __init__(self):
		self.points = np.zeros([par['batch_size'],2])#np.random.rand(par['batch_size'], 2)
		self.points_record = []
		self.vertex_record = []
		self.colors = []

	def next_vertices(self):

		if len(self.vertex_record) > (par['exclusion']-1) and par['exclusion'] > 0:
			
			v_prev = np.int32(np.array(self.vertex_record[-par['exclusion']-1:-1]))
			v_next = np.zeros(par['batch_size'])

			all_ids = np.arange(len(par['vertices']))
			for i in range(par['batch_size']):
				ids = np.setdiff1d(all_ids, (v_prev[:,i]+2)%len(par['vertices']), assume_unique=True)
				prs = softmax(par['likelihood'][ids])
				v_next[i] = np.random.choice(ids, p=prs)

			#pool = Pool(processes=4)
			#v_next = np.array(pool.map(vertex, [(i, v_prev[i]) for i in range(par['batch_size'])]))

			return np.int32(v_next)

		else:
			return np.random.choice(len(par['vertices']), size=par['batch_size'], p=par['probability'])


	def cast(self):
		vertex_id = self.next_vertices()
		next_vertex = np.array(par['vertices'])[vertex_id]
		cast_distance = np.array(par['distance'])[vertex_id,np.newaxis]

		
		self.points = self.points + cast_distance * (next_vertex - self.points)
		self.points_record.append(self.points)
		self.vertex_record.append(vertex_id)
		self.colors.append(np.array(par['colors'])[vertex_id])

	def recover_coords(self, transient=par['transient_iters']):

		points = np.concatenate(self.points_record, axis=0)
		points = points[par['transient_iters']:,:]

		colors = np.concatenate(self.colors, axis=0)
		colors = colors[par['transient_iters']:,:]

		return points[:,0], points[:,1], colors


def run_game():

	c = ChaosGame()
	for i in range(par['iterations']+1):
		if i%(par['iterations']//10)==0:
			print('Iteration {} of {}'.format(i, par['iterations']), end='\r')
		c.cast()

	xs, ys, cs = c.recover_coords()
	return xs, ys, cs
