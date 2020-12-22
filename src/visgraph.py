import tensorflow as tf

from model import ModelInpaint

# with tf.Session() as sess:
#     graph, graph_def = ModelInpaint.loadpb('../graphs/awesomegan-50k.pb', 'dcgan', awesome_gan=True)
#     writer = tf.summary.FileWriter("graphvis", graph)
#     #print(sess.run(h))
#     writer.close()

with tf.gfile.GFile('../graphs/awesomegan-50k.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='asdf')

    for op in graph.get_operations():
        print(op.name)

