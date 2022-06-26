from model import model
from losses import *
from dataset import Tensorflow_dataset
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

class ZERO_DCE:
	def __init__(self):
		self.dce_model = model()
		self.optimizer = tf.keras.optimizers.Adam()
		self.ill_w = 200
		self.color_w = 10

	def forward(self,data):
		output = self.dce_model(data)
		return output

	def get_enhanced_image(self, data, output):	
		enhanced_image = data
		split_d = tf.split(output,8,axis=3)
		for chunk in split_d:
			enhanced_image = enhanced_image + chunk * (enhanced_image - tf.square(enhanced_image))
		return enhanced_image	

	def compute_loss(self,data,output):
		enhanced_image = self.get_enhanced_image(data,output)
		losses = Loss()
		illumination_loss = self.ill_w * losses.illumination_smoothness_loss(output)
		spatial_loss = losses.spatial_loss(data,output)
		exp_loss = losses.exposure_control_loss(output)
		color_loss = self.color_w*losses.color_constancy_loss(output)
		total_loss = illumination_loss + spatial_loss + exp_loss + color_loss
		return total_loss

	def train(self,data,epochs=10):
		while epochs:
			output = self.forward(data)
			l_val = self.compute_loss(data,output)
			loss_val = lambda : self.compute_loss(data,self.forward(data))
			self.optimizer.minimize(loss_val,var_list=[self.dce_model.trainable_weights])
			tf.print(l_val)
			epochs -= 1	

	@tf.function
	def train(self,dataset,epochs=10):
		for e in range(epochs):
			print("Epoch : ",e+1)
			for step,(data) in enumerate(dataset):
				print("Step :",step+1)
				with tf.GradientTape() as tape:
					output = self.dce_model(data)
					loss = self.compute_loss(data,output)

				grads = tape.gradient(loss,self.dce_model.trainable_weights)
				self.optimizer.apply_gradients(zip(grads,self.dce_model.trainable_weights))

				tf.print('loss_val: ',loss)	

if __name__ == '__main__':
	train_data,test_data = Tensorflow_dataset('Dataset/LOLdataset/our485/low/',30).load_image()
	m = ZERO_DCE()
	m.train(train_data)