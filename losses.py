import tensorflow as tf

class Loss:

	def spatial_loss(self,actual_img,enhanced_img):
		left_kernel = tf.constant([[[[0,0,0]],[[-1,1,0]],[[0,0,0]]]],dtype=tf.float32)
		right_kernel = tf.constant([[[[0,0,0]],[[0,1,-1]],[[0,0,0]]]],dtype=tf.float32)
		up_kernel = tf.constant([[[[0,-1,0]],[[0,1,0]],[[0,0,0]]]],dtype=tf.float32)
		down_kernel = tf.constant([[[[0,0,0]],[[0,1,0]],[[0,-1,0]]]],dtype=tf.float32)
		
		act_img_mean = tf.reduce_mean(actual_img,axis=-1,keepdims=True)
		enh_img_mean = tf.reduce_mean(enhanced_img,axis=-1,keepdims=True)
		actual_img_avg = tf.nn.avg_pool2d(act_img_mean,ksize=4,strides=4,padding='VALID')
		enhance_img_avg = tf.nn.avg_pool2d(enh_img_mean,ksize=4,strides=4,padding='VALID')

		actual_left = tf.nn.conv2d(actual_img_avg,left_kernel,strides=1,padding='SAME')
		actual_right = tf.nn.conv2d(actual_img_avg,right_kernel,strides=1,padding='SAME')
		actual_up = tf.nn.conv2d(actual_img_avg,up_kernel,strides=1,padding='SAME')
		actual_down = tf.nn.conv2d(actual_img_avg,down_kernel,strides=1,padding='SAME')

		enhanced_left = tf.nn.conv2d(enhance_img_avg,left_kernel,strides=1,padding='SAME')
		enhanced_right = tf.nn.conv2d(enhance_img_avg,right_kernel,strides=1,padding='SAME')
		enhanced_up = tf.nn.conv2d(enhance_img_avg,up_kernel,strides=1,padding='SAME')
		enhanced_down = tf.nn.conv2d(enhance_img_avg,down_kernel,strides=1,padding='SAME')

		left = tf.square(actual_left-enhanced_left)
		right = tf.square(actual_right-enhanced_right)
		up = tf.square(actual_up-enhanced_up)
		down = tf.square(actual_down-enhanced_down)
		return tf.reduce_mean(left+right+up+down)

	def exposure_control_loss(self,enhanced_img,E=0.6):
		enhanced_img = tf.reduce_mean(enhanced_img,axis=-1,keepdims=True)
		pool_img = tf.nn.avg_pool2d(enhanced_img,ksize=16,strides=16,padding='VALID')
		return tf.reduce_mean(tf.math.abs(pool_img-E))

	def color_constancy_loss(self,enhanced_img):
		mean_val = tf.reduce_mean(enhanced_img,axis=(1,2),keepdims=True)	
		r,g,b = mean_val[:,:,:,0],mean_val[:,:,:,1],mean_val[:,:,:,2]
		return tf.reduce_mean(tf.square(r-g)+tf.square(r-b)+tf.square(b-g))

	def illumination_smoothness_loss(self,enhanced_img):
		batch_size = tf.shape(enhanced_img)[0]
		h_map = tf.shape(enhanced_img)[1]
		w_map = tf.shape(enhanced_img)[2]

		count_h = (tf.shape(enhanced_img)[2]-1)*tf.shape(enhanced_img)[3]
		count_w = tf.shape(enhanced_img)[2]*(tf.shape(enhanced_img)[3]-1)
		batch_size = tf.cast(batch_size,dtype=tf.float32)
		count_h = tf.cast(count_h,dtype=tf.float32)
		count_w = tf.cast(count_w,dtype=tf.float32)

		h_tv = tf.reduce_sum(tf.square((enhanced_img[:, 1:, :, :] - enhanced_img[:, : h_map - 1, :, :])))
		w_tv = tf.reduce_sum(tf.square((enhanced_img[:, :, 1:, :] - enhanced_img[:, :, : w_map - 1, :])))
		batch_size = tf.cast(batch_size, dtype=tf.float32)
		count_h = tf.cast(count_h, dtype=tf.float32)
		count_w = tf.cast(count_w, dtype=tf.float32)
		return 2 * (h_tv / count_h + w_tv / count_w) / batch_size