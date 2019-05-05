import tensorflow as tf

saver = tf.train.Saver()
meta_graph_def = tf.train.export_meta_graph(filename = '.models/my.meta')
