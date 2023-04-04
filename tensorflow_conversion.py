import keras
import tensorflow as tf
from keras import backend as K
import numpy as np

# Step 1: Load the Keras model
keras_model = keras.models.load_model('my_keras_model.h5')

# Step 2: Convert the Keras model to TensorFlow format
sess = K.get_session()
graph_def = sess.graph.as_graph_def()
tf_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, [keras_model.output.op.name])

# Step 3: Optimize the TensorFlow graph using SageMaker Neo's TensorFlow API
optimized_graph = tf.compat.v1.neo.optimizing.optimize_graph(tf_graph)

# Step 4: Save the optimized TensorFlow graph as a .pb file
tf.io.write_graph(optimized_graph, '.', 'my_tensorflow_model.pb', as_text=False)

# Step 5: Test the optimized TensorFlow model
with tf.compat.v1.gfile.FastGFile('my_tensorflow_model.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

x_test = np.random.rand(1, 224, 224, 3)
output_tensor = sess.graph.get_tensor_by_name('import/' + keras_model.output.name + ':0')
result = sess.run(output_tensor, feed_dict={'import/input_1:0': x_test})
print(result)
