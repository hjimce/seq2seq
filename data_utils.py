#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0#为了batch_size训练，而采用补齐方案
GO_ID = 1
EOS_ID = 2
UNK_ID = 3#未登陆、低频过滤词的id号


#当读取一个训练数据的每一行的时候,我们接着根据一些标点符号,进行切割成子句子,比如逗号\分号等
#具体请看_WORD_SPLIT里面的标点符号
def split_english(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(b"([.,!?\"':;)(])", space_separated_fragment))
    for l in words:
        print (l)
    return [w for w in words if w]
def split_chinese(sentence):
    line=re.split(u'【（。，！？、“：；）】',sentence.decode('utf-8'))
    for l in line:
        print (l)
    ws=[]
    for words in line:
        for w in words:
            ws.append(w)
    return  ws


#创建词典,如果vocabulary_path为空,那么根据data_path读取数据,然后根据词频统计,求取前max_vocabulary_size个词,作为词典
#低频、未登录的词都没有加入词典，它们后面统一用同一个id号：UNK_ID
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,ischinese=False):
  #根据训练数据,创建词典

    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
        for line in f:
          #
            if ischinese:
                tokens=split_chinese(line)
            else:
                tokens = split_english(line)

            for w in tokens:
                word =w# re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:

                vocab_file.write(w.encode('utf-8') + b"\n")
  #else:#如果是中文的话,那么是每个字一个切分的



def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,ischinese):
    if ischinese:
        words = split_chinese(sentence)
    else:
        words = split_english(sentence)
    #如果指定的键值w不存在，那么就返回UNK_ID
    return [vocabulary.get(w.encode('utf-8') , UNK_ID) for w in words]




def data_to_token_ids(data_path, target_path, vocabulary_path,ischinese):
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
        with gfile.GFile(target_path, mode="w") as tokens_file:
            for line in data_file:
                token_ids = sentence_to_token_ids(line, vocab, ischinese)
                tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir,chi_vocabulary_size, eng_vocabulary_size, tokenizer=None):
  engdata_path ='data/english.utf8'
  chidata_path ='data/chinese.utf8'

  # 创建词典,并保存到相应的文件夹
  eng_vocab_path ='data/engvoca.utf8'
  chi_vocab_path ='data/chivoca.utf8'
  create_vocabulary(eng_vocab_path, engdata_path, eng_vocabulary_size, False)
  create_vocabulary(chi_vocab_path, chidata_path, chi_vocabulary_size, True)

  #根据上面已经保存好的词典文件,把
  eng_train_ids_path ='data/engid.utf8'
  chi_train_ids_path ='data/chiid.utf8'
  data_to_token_ids(engdata_path, eng_train_ids_path, eng_vocab_path, False)
  data_to_token_ids(chidata_path, chi_train_ids_path, chi_vocab_path, True)



  return (eng_train_ids_path, chi_train_ids_path,
          eng_train_ids_path, chi_train_ids_path,
          eng_vocab_path, chi_vocab_path)
