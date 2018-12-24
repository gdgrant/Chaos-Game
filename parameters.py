import numpy as np

global par
par = {
	# 'vertices'			:	[[0,0],[1,0],[0.5,1]],
	# 'likelihood'		:	[1,1,1],
	# 'distance'			:	[0.55, 0.35, 0.45],
	# 'colors'			:	[[0.9,0.1,0.1],[0.1,0.9,0.1],[0.1,0.1,0.9]],

	# 'vertices'			:	[[0,0],[1,0],[0,1],[1,1]],
	# 'likelihood'		:	[1,1,1,1],
	# 'distance'			:	[0.35, 0.5, 0.5, 0.35],
	# 'colors'			:	[[0.3,0.8,0.8],[0.6,0.1,0.8],[0.2,0.1,0.8],[0.5,0.0,0.3]],

	'vertices'			:	[[0,0.5],[0.5,0],[1,0.5],[0.5,1],[0.5,0.5]],
	'likelihood'		:	[1,0.1,1,1,0.1],
	'distance'			:	[0.5, 0.6, 0.4, 0.5, 0.25],
	'colors'			:	[[0.3,0.8,0.8],[0.6,0.1,0.8],[0.2,0.1,0.8],[0.5,0.0,0.3],[0.5,0.5,0.5]],


	'exclusion'			:	2,

	'batch_size'		:	50,
	'iterations'		:	10000,
	'transient_iters'	:	1000,

	'image_size'		:	[int(1.2*1920),int(1.2*1080)]
}


def softmax(x):
	return np.exp(x)/(np.sum(np.exp(x)))

assert len(par['vertices'])==len(par['likelihood'])
assert len(par['likelihood'])==len(par['distance'])
assert len(par['colors'])==len(par['distance'])

par['likelihood'] = np.array(par['likelihood'])
par['probability'] = softmax(par['likelihood'])

par['exclusion'] = par['exclusion'] if len(par['vertices']) > 3 else 0
