import numpy as np
import torch

from scipy.ndimage import zoom

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, *state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, *state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	

def pre_process_img(img):
	# h, w, c -> c, h, w
    img = img.transpose((2, 0, 1))
    
	# resize image
	# # large image is shape (3, 480, 480)
	# # small image is shape (3, 60, 60)
    # input_size = 480	
    # output_size = 60
    # bin_size = input_size // output_size
    # small_image = img.reshape((3, output_size, bin_size, 
	# 									output_size, bin_size)).max(4).max(2)
    small_image = zoom(img, (1, 0.15, 0.15))
    small_image = torch.FloatTensor(small_image).to(DEVICE)
    return small_image.unsqueeze(0)
