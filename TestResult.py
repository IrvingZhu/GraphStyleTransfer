import tensorflow as tf
import transformNet
import numpy as np
import cv2
import os

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

MEAN_PIXEL=np.array([123.68,116.779,103.939])

#default
img_length=256
img_width=256
batch_size=1

current_dir='./MachLearning/tensorflow/SelfPerceptual'+'/'
save_model_dir=current_dir+'modelFile'
test_con_img=current_dir+'test_dir'
save_dir=current_dir+'test_result/'

def preprocess(input_img):
    return input_img-MEAN_PIXEL

def unpreprocess(input_img):
    return input_img+MEAN_PIXEL

def load_img(save_dir,i=0,resize=True,process=True):
    img_name=os.listdir(save_dir)
    if i == '':
        return None
    i = int(i)
    if i > len(img_name):
        return None
    img_path=save_dir+'/'+img_name[i]
    return_img=cv2.imread(img_path)
    if resize==True:
        return_img=cv2.resize(return_img,(img_length,img_width))
    if process==True:
        return_img=preprocess(return_img)
    return return_img

def save_img(save_dir,input_img,unprocess=True):
    if unprocess==True:
        input_img=unpreprocess(input_img)
    input_img=np.clip(input_img,0,255).astype(np.uint8)
    res_dir=input("input the storage direction:")
    save_dir=save_dir+res_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    is_content=input("Is input picture content?[y/n]")
    if is_content=='y':
        save_dir=save_dir+'/content.jpg'
    else:
        save_dir=save_dir+'/test.jpg'
    cv2.imwrite(save_dir,input_img[0])

def main():
    in_img=tf.placeholder(tf.float32,shape=(batch_size,img_length,img_width,3),name='input')
    op_img=transformNet.ImageNet(in_img)
    with tf.Session() as sess:
        img_number=input("which picture should you load? [number]")        
        test_img=load_img(test_con_img,i=img_number)
        #model_name=input("which model you want to load? [all of name]")
        #saver=tf.train.import_meta_graph(model_name)

        cpkt_save_dir=input("Please input the cpkt direction... [your storage direction]")
        cpkt_save_name=input("Please input the cpkt name... [your model name]")
        saver=tf.train.Saver()
        saver.restore(sess=sess,save_path=cpkt_save_dir+'/'+cpkt_save_name+'.cpkt')

        #Input_img=sess.graph.get_tensor_by_name('c_img_ph:0')
        #op=sess.graph.get_tensor_by_name('out_img:0')

        test_img=np.reshape(test_img,(-1,img_length,img_width,3))
        res=op_img.eval(feed_dict={in_img:test_img})

        save_img(save_dir,res)
        save_img(save_dir,test_img)

    print("the test program is finished!")    

main()