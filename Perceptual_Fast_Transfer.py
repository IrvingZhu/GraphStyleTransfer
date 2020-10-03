import tensorflow as tf
import transformNet
import numpy as np
import scipy.io
import cv2
import scipy.misc
import os
import functools
from operator import mul

MEAN_PIXEL = np.array([123.68, 116.779, 103.939])

#picture imformation and loss parameter
#learning imformation
learning_rate = 0.001
train_epoch = 2
img_length = 256
img_width = 256
alpha = 64
beta = 8
tv_weight = 16

#get current dir
vgg19_path = './MachLearning/tensorflow/SelfPerceptual/imagenet-vgg-verydeep-19.mat'
current_dir = './MachLearning/tensorflow/SelfPerceptual'+'/'
#current_dir=os.getcwd()+'/'
save_dir = current_dir+'content_train/train2014_mini'
style_img_path = current_dir+'style_train'
# res_path=current_dir+'output_graph'
test_dir = current_dir+'test_dir'
model_save_dir = current_dir+'modelFile/'
output_test_dir = current_dir+'output_test_dir/test_out.jpg'
#noise=0.5
batch_size = 1

#batch_shape
batch_shape = (batch_size, img_length, img_width, 3)

#VGG19 output layer
style_layers = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2),('conv4_1', 0.2), ('conv5_1', 0.2)]  # name and weight
content_layers = [('conv4_2', 1)]  # also


def load_img(save_dir, i=0, img_pre=True):
    img_name = os.listdir(save_dir)
    if i == '':
        return None
    i = int(i)
    if i >= len(img_name):
        return None
    img_path = save_dir+'/'+img_name[i]
    return_img = cv2.imread(img_path)
    return_img = cv2.resize(return_img, (img_length, img_width))
    if img_pre == True:
        return_img = preprocess(return_img)
    return return_img


def save_img(input_img, end_path):
    input_img = unpreprocess(input_img)
    input_img = np.clip(input_img, 0, 255).astype(np.uint8)
    cv2.imwrite(end_path, input_img[0])


def preprocess(input_img):
    return input_img-MEAN_PIXEL


def unpreprocess(input_img):
    return input_img+MEAN_PIXEL


def vgg19Model(data_path, input_img):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
    vgg19 = scipy.io.loadmat(data_path)
    weight = vgg19['layers'][0]  # every layers weight

    net = {}
    current = input_img
    net['input'] = current
    for i, name in enumerate(layers):
        layer_name = name[0]
        if layer_name == 'c':  # convolution
            kernels, bias = weight[i][0][0][0][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = m_conv_2d(current, kernels, bias)
        elif layer_name == 'p':  # pooling
            current = m_pool_2d(current)
        elif layer_name == 'r':  # relu activating
            current = tf.nn.relu(current)
        net[name] = current
    return net


def m_conv_2d(input, weight, bias):
    conv = tf.nn.conv2d(input, tf.constant(weight),strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


def m_pool_2d(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def gramMatrix(input_matrix_img, size, deep):
    input_matrix_img = np.reshape(input_matrix_img, (size, deep))
    print("input_matrix_img.shape:", input_matrix_img.shape)
    g_matrix = np.matmul(input_matrix_img.T, input_matrix_img)/input_matrix_img.size
    print("gram shape ", g_matrix.shape)
    return g_matrix


def get_style_features(sess, style_img):
    style_features = {}
    style_shape = (1,)+style_img.shape

    style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
    style_image_pre = preprocess(style_image)

    net = vgg19Model(vgg19_path, style_image_pre)
    style_pre = np.array([style_img])

    for layer, _ in style_layers:
        features = net[layer].eval(feed_dict={style_image: style_pre})
        print("features.shape:", features.shape)
        gram = gramMatrix(features, -1, features.shape[3])
        style_features[layer] = gram

    return style_features


def get_content_features(sess, content_img):
    # precompute content features
    content_features = {}
    content_net = vgg19Model(vgg19_path, content_img)

    for layer, _ in content_layers:
        content_features[layer] = content_net[layer]
        print("content_features[", layer, "].shape:",content_features[layer].shape)

    return content_features


def content_loss(sess, content_weight, content_features, net):
    content_loss = 0.0
    for name, _ in content_layers:
        t_size = tensor_Size(content_features[name], mul)
        content_size = t_size*batch_size
        assert tensor_Size(content_features[name], mul) == tensor_Size(net[name], mul)
        content_loss = 2.0 * tf.nn.l2_loss(net[name]-content_features[name])/content_size.value

    content_loss /= len(content_layers)  # the average loss
    content_loss = content_weight*content_loss
    return content_loss


def style_loss(sess, style_weight, style_features, net):
    style_losses = []
    for style_layer, _ in style_layers:
        layer = net[style_layer]
        bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
        size = height * width * filters
        feats = tf.reshape(layer, (bs, height * width, filters))
        feats_T = tf.transpose(feats, perm=[0, 2, 1])
        grams = tf.matmul(feats_T, feats) / size
        style_gram = style_features[style_layer]
        style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

    style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size
    return style_loss


def total_variable_denoising(input_img, tv_weight, mul):
    tv_y_size = tensor_Size(input_img[:, 1:, :, :], mul)
    tv_x_size = tensor_Size(input_img[:, :, 1:, :], mul)
    y_tv = tf.nn.l2_loss(input_img[:, 1:, :, :] - input_img[:, :input_img.shape[1]-1, :, :])
    x_tv = tf.nn.l2_loss(input_img[:, :, 1:, :] - input_img[:, :, :input_img.shape[2]-1, :])
    # total variable normalization loss
    tot_var_deno_loss = tv_weight * 2 * (x_tv/tv_x_size.value + y_tv/tv_y_size.value)/batch_size
    return tot_var_deno_loss


def tensor_Size(input_img, mul):
    return functools.reduce(mul, (d for d in input_img.shape[1:]), 1)


def main():
    #load style_img
    name_style=input("Which picture you will use? please input the number...")
    style_img = load_img(style_img_path,i=name_style)

    #define some important variable
    batch_shape = (batch_size, img_length, img_width, 3)

    c_img_ph = tf.placeholder(tf.float32, shape=batch_shape,name='c_img_ph')
    #model
    out_img = transformNet.ImageNet(c_img_ph/255.0)
    #out_img_pre=preprocess(out_img)
    net = vgg19Model(vgg19_path, out_img)

    #builder=tf.saved_model.builder.SavedModelBuilder(model_save_dir)
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            style_features = get_style_features(sess, style_img)
            content_features = get_content_features(sess, c_img_ph)

            all_loss = content_loss(sess, alpha, content_features, net)+style_loss(sess, beta, style_features, net)+total_variable_denoising(out_img, tv_weight, mul)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(all_loss)

            list_cont_dir = os.listdir(save_dir)
            train_num = len(list_cont_dir)//batch_size

            sess.run(tf.global_variables_initializer())

            #one code
            epoch = 0
            for epoch in range(train_epoch):
                i = 0
                print("train epoch is ", epoch+1, " lets begin")
                for i in range(train_num-1):
                    con_img = load_img(save_dir, i)
                    con_img = np.reshape(con_img, (-1, img_length, img_width, 3))

                    _, loss = sess.run([optimizer, all_loss], feed_dict={c_img_ph: con_img})
                    print("train step is:", i, "------loss value is:", loss)

            #another code
            '''
            i=0     
            for i in range(train_num-1):
                con_img=load_img(save_dir)
                con_img=np.reshape(con_img,(-1,img_length,img_width,3))
                print("train num is ",i+1," ----------------")
                for epoch in range(train_epoch):
                    _,loss=sess.run([optimizer,all_loss],feed_dict={c_img_ph:con_img})
                    print("    train epoch is ",epoch+1," the loss is ",loss)
            '''

        #model test
        test_img_num = input("which content picture you will use? please input the number...")
        test_img = load_img(test_dir, i=test_img_num, img_pre=False)
        test_img = np.reshape(test_img, (-1, img_length, img_width, 3))
        img = out_img.eval(feed_dict={c_img_ph: test_img})
        save_img(img, output_test_dir)

        #model save
        # comment: the model cannot use GPU to save,should put them out of gpu restrict
        bool_save = input("the picture is saved in " + output_test_dir + "\r\n" + "the training is finished,Do you want to save it? [y/n]")
        if bool_save == 'y':
            yourname = input("please input your model name")
            tf.train.Saver().save(sess=sess, save_path=model_save_dir + yourname + '/' + yourname + '.cpkt')

    #finished
    print("Successful transfer the style,next exit the program immediately...")
    #builder.save()


main()

'''
出错总结:
    1.没有正确理解函数传参以及返回值的基本原理
    2.梯度下降的作用是更新W和参数b,因此只需要在第一次参数设置的时候设置一个W初始值就行了
    3.tensor值是不能被传进优化器里面的，否则会产生意想不到的结果
    4.sess.run(xxx),xxx为节点，节点输出值，首先要在参数里面注入feed_dict，否则会报错
'''