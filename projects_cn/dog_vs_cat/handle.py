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
import time
import numpy as np
import tensorflow as tf
from predict import create_inception_resnet_v2
from keras.preprocessing import image

dict={'0':'猫','1':'狗'}
graph = tf.get_default_graph()
with graph.as_default():
	model = create_inception_resnet_v2()
	model.load_weights('tf-cat-dog-weight-normal.h5')

def predict(imgpath) :
    img = image.load_img(imgpath, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    with graph.as_default():
    	preds = model.predict(x)
    a=np.argmax(preds)
    print(a)
    return dict[str(a)]

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
