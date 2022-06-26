import os
import tensorflow as tf

def check_images( s_dir):
    bad_images = []
    for i in os.listdir(s_dir):
        try:
        	file_E = tf.io.read_file(s_dir+i)
        	img = tf.io.decode_png(file_E)
        except:
            os.remove(s_dir+i)    
check_images('./Dataset/LOLdataset/our485/low/')
