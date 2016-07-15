#coding=utf-8
from tensorflow.python.platform import gfile
import  re

with open('data/chinese.utf8', mode="rb") as f:
    for line in f:
        p=re.split(u'，|。',line.decode('utf-8'))
        for l in p:
            print l
