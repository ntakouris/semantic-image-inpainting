import tensorflow as tf

from model import ModelInpaint

with tf.Session() as sess:
    graph, graph_def = ModelInpaint.loadpb('../graphs/dcgan-100.pb', 'dcgan')
    writer = tf.summary.FileWriter("graphvis", graph)
    #print(sess.run(h))
    writer.close()