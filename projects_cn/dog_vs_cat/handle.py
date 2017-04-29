#!/usr/bin/python
# -*- coding: utf-8 -*-
# filename: handle.py
import hashlib
import reply
import receive
import web
import requests
import urlparse
from wechatpy import parse_message
from wechatpy.replies import create_reply, ArticlesReply
import cv2
import time
import numpy as np
import tensorflow as tf
from keras.applications import *
from keras.models import *
from keras.layers import *

from bottleneck_ir2 import create_inception_resnet_v2
from keras.preprocessing import image

def model(MODEL,width=299,height=299,lambda_func=None):
    x= Input((height, width, 3))
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model=MODEL(input_tensor=x, weights='imagenet', include_top=False)
    return Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

def data_input(imgpath,width=299,height=299):
    img= image.load_img(path=imgpath, target_size=(height, width))
    return np.expand_dims(image.img_to_array(img), axis=0)
def input_n(imgpath,model_name,width=299,height=299,lambda_func=None):
    format_input=data_input(imgpath,width,height)
    m=model(model_name,width,height,lambda_func)
    return m.predict(format_input)
graph = tf.get_default_graph()
with graph.as_default():
        m_ir2 = create_inception_resnet_v2()
        m_ir2.load_weights('weight-v1.h5',by_name=True)
        m=Model(m_ir2.input, GlobalAveragePooling2D()(m_ir2.output))
        m_resnet=model(ResNet50,224,224)
        #m_xecpiton=model(Xception,299,299,xception.preprocess_input)
        model=load_model('model_bottleneck_product.h5')
        print('init done')
def predict(imgpath) :
    input_f1=data_input(imgpath,224,224)
    #input_f2=data_input(imgpath,299,299)
    img= image.load_img(path=imgpath, target_size=(299, 299))
    input_f3=np.expand_dims(np.asarray(img,dtype="float32")/255, axis=0)
    with graph.as_default():
        input1=m_resnet.predict(input_f1)
        #input2=m_xecpiton.predict(input_f2)
        input3=m.predict(input_f3)
        x_input=[]
        x_input.append(np.array(input1))
        #x_input.append(np.array(input2))
        x_input.append(np.array(input3))
        x_input = np.concatenate(x_input, axis=1)

        pred=model.predict(x_input)
    rtn='无法识别这是什么'
    if pred[0]>0.55 :
        rtn='狗'
    elif pred[0]<0.45:
        rtn='猫'
    print(rtn)
    return rtn

class Handle(object):
    def POST(self):
        webData = web.data()
        print "Handle Post webdata is ", webData   #后台打日志
        msg = parse_message(webData)
        reply = ''
        try:
            if msg.type == 'text':
                reply = create_reply('Text:' + msg.content.encode('utf-8'), message=msg)
            elif msg.type == 'image':
                reply = create_reply('图片', message=msg)
                r = requests.get(msg.image) # download image
                filename = 'static/img/' + str(int(time.time())) + '.jpg';
                convertfilename = filename.replace('.', '.convert.')
                with open(filename, 'w') as f:
                    f.write(r.content)
                if cv2.imread(filename) is not None:
                    reply = create_reply(predict(filename), message=msg)
        except Exception, e:
            reply = create_reply('识别失败', message=msg)
            print e
        xml=reply.render()
        return xml
    def GET(self):
        try:
            data = web.input()
            if len(data) == 0:
                return "hello, this is handle view"
            signature = data.signature
            timestamp = data.timestamp
            nonce = data.nonce
            echostr = data.echostr
            token = "william" #请按照公众平台官网\基本配置中信息填写

            list = [token, timestamp, nonce]
            list.sort()
            sha1 = hashlib.sha1()
            map(sha1.update, list)
            hashcode = sha1.hexdigest()
            print "handle/GET func: hashcode, signature: ", hashcode, signature
            if hashcode == signature:
                return echostr
            else:
                return ""
        except Exception, Argument:
            return Argument
