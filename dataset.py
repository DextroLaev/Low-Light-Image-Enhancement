import tensorflow as tf
import os
tfds = tf.data.Dataset

class Tensorflow_dataset:
	def __init__(self,path,batch_size=100,test_split=0.2):
		self.path = path
		self.batch_size = batch_size
		self.total_files = 485
		self.train_size = int(self.total_files*(1-test_split))
		self.test_size = int(self.total_files*test_split)

	def get_image(self,file):
		img = tf.io.read_file(file)
		decode_png = tf.io.decode_png(img,channels=3)
		decode_png = tf.image.resize(decode_png,[300,300])
		return decode_png/255.0

	def preprocess_img(self,img):
		return img/255.0	

	def parse_func(self,data,dataset_type):
		if dataset_type == 'batched_data':
			img=[]
			for i in range(len(data)):
				img.append(self.get_image(data[i]))
			img = tf.cast(img,dtype=tf.float32)
		elif dataset_type == 'stream_data':
			img = tf.expand_dims(self.get_image(data),axis=0)
		return img

	def load_image(self):
		data = tfds.list_files(self.path+'*',shuffle=True)
		train_ds = data.take(self.train_size)
		test_ds = data.skip(self.train_size).take(self.test_size)

		train_ds_batch = train_ds.batch(self.batch_size,drop_remainder=True)
	
		train_data = train_ds_batch.map(lambda x:self.parse_func(x,'batched_data')).map(self.preprocess_img)
		test_data = test_ds.map(lambda x:self.parse_func(x,'stream_data')).map(self.preprocess_img)
		return train_data,test_data