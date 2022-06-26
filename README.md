# Low-Light-Image-Enhancement
Low-light-Image enhancement based on paper ZERO-DCE.

Dataset used for training and testing : [LOL-Dataset](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view)

Make sure your folder looks like this:

![Alt text](Img/Screenshot_2022-06-26_21-34-48.png?raw=true "Title")

### For Training and Testing follow the given procedure:
---------------

   The below code helps to remove the unnecessary/corrupted files from dataset folder  
    
    $ python check.py
   
   Train the model and save the weights
   
    $ python keras_api_train.py
   
   There are 2 training files, ```Tensorflow_api_train.py``` file is only for prototyping purpose (NOT FOR PRACTICAL USE) 
   
   Testing the model on the given dataset
   
    $ python test.py

### Research paper used
----
  
  [Zero-reference for low light enhancement](https://arxiv.org/pdf/2001.06826.pdf)
  
  [Learning to Enhance Low-Light Imagevia Zero-Reference Deep Curve Estimation](https://arxiv.org/pdf/2103.00860.pdf)
