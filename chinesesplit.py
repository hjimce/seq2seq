from tensorflow.python.platform import gfile
import  re

with gfile.GFile('data/chinese.utf8', mode="rb") as f:
    for line in f:
        line=re.split(':',line)[0]
        print line

