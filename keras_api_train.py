import tensorflow as tf
from model import *
from dataset import *
from losses import *

class Zero_Dce(tf.keras.Model):

	def __init__(self):
		super(Zero_Dce,self).__init__()
		self.dce_model = model()
		self.optimizer = tf.keras.optimizers.Adam()
		self.ill_w = 1
		self.color_w = 1
		self.exp_w = 1

	def get_enhanced_image(self,data,output):
		enhanced_img = data
		split_d = tf.split(output,8,axis=3)
		for chunk in split_d:
			enhanced_img += chunk * (enhanced_img-tf.square(enhanced_img))
		return enhanced_img

	def custom_loss(self,data,output):
		enhanced_img = self.get_enhanced_image(data,output)
		losses  = Loss()
		illumination_loss = self.ill_w * losses.illumination_smoothness_loss(output) 
		spatial_loss = losses.spatial_loss(data,output)
		exp_loss = self.exp_w*losses.exposure_control_loss(output)
		color_loss = self.color_w*losses.color_constancy_loss(output)
		total_loss = illumination_loss + spatial_loss + exp_loss + color_loss
		return {'total_loss':total_loss}

	def call(self,data):
		output = self.dce_model(data)
		return self.get_enhanced_image(data,output)

	def train_step(self,data):
		with tf.GradientTape() as tape:
			output = self.dce_model(data)
			losses = self.custom_loss(data,output)
		grads = tape.gradient(losses,self.dce_model.trainable_weights)
		self.optimizer.apply_gradients(zip(grads,self.dce_model.trainable_weights))
		return losses

	def test_step(self,data):
		output = self.dce_model(data)
		return self.custom_loss(data,output)

	def save(self,filepath,overwrite=True):
		folder_exist = os.path.isdir('Model')
		if not folder_exist:
			os.mkdir('Model')
		self.dce_model.save_weights('Model'+'/'+filepath,overwrite)

	def load(self,filepath):
		self.dce_model.load_weights(filepath)

	def predict(self,img):
		img = tf.cast(img,dtype=tf.float32)/255.0
		img = tf.image.resize(img,[300,300])
		img = tf.expand_dims(img,axis=0)
		out_img = self.call(img)
		out_img = out_img[0,:,:,:]*255
		return tf.cast(out_img,dtype=tf.uint8)

if __name__ == '__main__':
	train_data,test_data = Tensorflow_dataset('Dataset/LOLdataset/our485/low/',30).load_image()
	m = Zero_Dce()
	m.compile()
	m.fit(train_data.prefetch(tf.data.AUTOTUNE),validation_data=test_data.prefetch(tf.data.AUTOTUNE),epochs=10)
	m.save_model('my_model')