import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

model = tf.compat.v1.saved_model.load_keras_model('data/trt_fp16_dir/saved_model')

# Convert the model to a TensorRT optimized graph
converter = trt.TrtGraphConverterV2(input_saved_model_dir='data/trt_fp16_dir/saved_model',
                                    conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS)
converter.convert()

# Save the optimized graph to a new directory
converter.save('data/trt_fp16_dir/optimized_model')