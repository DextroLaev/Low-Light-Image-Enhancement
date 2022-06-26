import tensorflow
from dataset import *
from keras_api_train import Zero_Dce
import matplotlib.pyplot as plt

def test(model,path):
	for i in os.listdir(path):
		fileE = tf.io.read_file(path+'/'+i)
		img = tf.io.decode_png(fileE,channels=3)
		out_img = model.predict(img)
		plt.imshow(out_img)
		plt.show()

if __name__ == '__main__':
	model = Zero_Dce()
	model.load('Model/my_model')
	test_path = './Dataset/LOLdataset/eval15/low/'	
	test(model,test_path)
