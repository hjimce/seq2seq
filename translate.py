#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils
import seq2seq_model


'''tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 400, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 400, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS'''
en_vocab_size=400
ch_vocab_size=400


#采用pad的方式,主要是为了batch训练,提高训练效率,(5,10)表示输入序列batch的长度全部为5,输出序列为10
#当一个英文句子进来的时候,我们首先判断它的长度,属于哪个buckets,然后在进行pad补齐
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(40, 50)]

#逐行读取训练数据，并根据句子的长度把它存储到data_set中，返回data_set
#比如data_set[2][3][1]就表示源语言句子长度位于5~10,目标语言句子长度位于10~15；第三个符号要求的句子，最后一维[1]表示目标语言
def read_data(source_path, target_path):

  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      #逐行读取(逐句),每一个英语句子对应一个目标语言一个句子
      source, target = source_file.readline(), target_file.readline()
      max_size=20#当读取了句子达到这个数,就结束(用于代码调试,否则每次都要全部读取,那就死人了)
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1

        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)#目标语言,在输出每个句子后面都要加入一个结束标识符,encoder-decoder要使用
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])#在每个bucket里面,存放了各自长度集合的句子
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

#创建模型
def create_model(session, forward_only):
  #1024表示隐藏层神经元的个数，3表示网络共三层
  numberhidd=1024
  numlayer=3
  max_gradient_norm=5.#RNN防止梯度爆炸，所以需要在训练的时候，加入梯度裁剪
  batch_size=5
  learning_rate=0.5
  learning_rate_decay_factor=0.99




  model = seq2seq_model.Seq2SeqModel(en_vocab_size,ch_vocab_size , _buckets,numberhidd,numlayer,
                                     max_gradient_norm, batch_size,learning_rate, learning_rate_decay_factor,forward_only=forward_only)
  #如果已经有训练好的模型,那么直接加载参数,否则就初始化全部的参数
  '''ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:'''
  session.run(tf.initialize_all_variables())
  return model

#训练函数
def train():
  #创建词典，最后返回训练数据id映射文件
  en_train, ch_train, _, _ = data_utils.prepare_wmt_data(400,400)

  with tf.Session() as sess:
    model = create_model(sess, False)

    dev_set = read_data(en_train, ch_train)#测试使用的数据
    train_set = read_data(en_train, ch_train)#返回的数据句子，还没经过pad补齐
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]#保存了每个bucket中，句子的个数
    #print (train_set[2])
    train_total_size = float(sum(train_bucket_sizes))#训练数据总共有多少个句子


    #这个是为了合理分配每个bucket中，训练的时候batchsize的大小选择问问题，选择概率用的
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # 开始循环训练
    print ('…………………………………………开始训练×××××××××')
    while True:
      #每次训练，我们都从所有的bucket中，随机选一个bucket(根据bucket句子个数，句子多的，选中的概率大)
      #然后从选中的bucket中，我们又随机的选出batch个句子，进行训练
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])
      print (bucket_id)
      #获取batch训练数据
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      print (encoder_inputs)
      print (decoder_inputs)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, False)




      #验证阶段，每训练n次，我们就验证一次，打印结果
      '''if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()'''

#预测阶段
def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  #预测阶段我们只输入一个句子

    # 加载词汇表
    en_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.en" % FLAGS.en_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.fr" % FLAGS.fr_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # 翻译:我们用控制台输入英语句子
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    #对输入结果进行翻译解码
    while sentence:
      # 先把输入的单词,转换成索引形式
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
      # 根据句子的长度,判读属于哪个buckets
      bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])

      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      #得到概率输出序列
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      #取一个输出序列的argmax最大的概率单词
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

      if data_utils.EOS_ID in outputs:#如果翻译结果中存在EOS_ID,那么我们只需截取前面的单词,作为结果
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # 大印结果
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


train()
'''def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()'''
