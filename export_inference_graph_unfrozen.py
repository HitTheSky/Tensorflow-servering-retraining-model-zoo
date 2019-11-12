"""
Run this with
    cd research/object_detection
    python export_inference_graph_unfrozen.py --input_type encoded_image_string_tensor --pipeline_config_path samples/configs/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix ssd_mobilenet_v1_coco_2017_11_17/model.ckpt --output_directory /Users/erotundo/HAL24K/Projects/dockerfiles-tf-serving/ssd_mobilenet_v1_coco/models
Make sure you add object_detection in PYTHONPATH
Credits to original gist by @dnlglsn: 
https://gist.github.com/dnlglsn/c42fbe71b448a11cd72041c5fcc08092
References:
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/exporting_models.md
https://github.com/tensorflow/models/issues/1988
Unfortunately, the tutorial for saving a model for inference "freezes" the
variables in place and makes them unservable by tensorflow_serving.
export_inference_graph.py exports an empty "variables" directory, which needs to
be populated.
The below script, which is a modified version of export_inference_graph, will
save the model in an "unfrozen" state, servable via TensorFlow Serving.
"""

import logging
import os
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.exporter import (input_placeholder_fn_map,
                                       _add_output_tensor_nodes,
                                       write_graph_and_checkpoint)


flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`]')
flags.DEFINE_string('pipeline_config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_prefix', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', None, 'Path to write outputs.')
flags.DEFINE_string('version', None, 'Version of the output model')
flags.DEFINE_string('model_name', None, 'Model name and version (model string) of the output model')

FLAGS = flags.FLAGS

def _write_saved_model(saved_model_path,
                       trained_checkpoint_prefix,
                       inputs,
                       outputs):
    """Writes SavedModel to disk.
    Args:
      saved_model_path: Path to write SavedModel.
      trained_checkpoint_prefix: path to trained_checkpoint_prefix.
      inputs: The input image tensor to use for detection.
      outputs: A tensor dictionary containing the outputs of a DetectionModel.
    """
    # Optimizing graph
    rewrite_options = rewriter_config_pb2.RewriterConfig()
    rewrite_options.optimizers.append('pruning')
    rewrite_options.optimizers.append('constfold')
    rewrite_options.optimizers.append('layout')
    graph_options = tf.GraphOptions(rewrite_options=rewrite_options, infer_shapes=True)

    # Build model for TF Serving
    config = tf.ConfigProto(graph_options=graph_options)

    # @TODO: add XLA for higher performance (AOT for ARM, JIT for x86/GPUs)
    # https://www.tensorflow.org/performance/xla/
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    saver = tf.train.Saver()
    with session.Session(config=config) as sess:
        # Restore variables from training checkpoints
        saver.restore(sess, trained_checkpoint_prefix)

        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

        tensor_info_inputs = {'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
        tensor_info_outputs = {}
        for k, v in outputs.items():
            tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

        detection_signature = (
                    tf.saved_model.signature_def_utils.build_signature_def(
                            inputs     = tensor_info_inputs,
                            outputs    = tensor_info_outputs,
                            method_name= signature_constants.PREDICT_METHOD_NAME)
        )

        builder.add_meta_graph_and_variables(
                    sess, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={'predict_images': detection_signature,
                                       signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: detection_signature,
                                       },
            )
        builder.save()


def _export_inference_graph(input_type,
                            detection_model,
                            use_moving_averages,
                            trained_checkpoint_prefix,
                            output_directory,
                            optimize_graph=False,
                            output_collection_name='inference_op'):
    """Export helper."""
    tf.gfile.MakeDirs(output_directory)
    saved_model_path = os.path.join(output_directory, FLAGS.model_name, str(FLAGS.version))

    if input_type not in input_placeholder_fn_map:
        raise ValueError('Unknown input type: {}'.format(input_type))
    placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type]()
    inputs = tf.to_float(input_tensors)
    preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)
    output_tensors = detection_model.predict(preprocessed_inputs, true_image_shapes)
    postprocessed_tensors = detection_model.postprocess(output_tensors, true_image_shapes)
    outputs = _add_output_tensor_nodes(postprocessed_tensors,
                                       output_collection_name)

    saver = None
    if use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver()


    _write_saved_model(saved_model_path,
                       trained_checkpoint_prefix,
                       inputs,
                       outputs)


def export_inference_graph(input_type,
                           pipeline_config,
                           trained_checkpoint_prefix,
                           output_directory,
                           optimize_graph=False,
                           output_collection_name='inference_op'):
    """Exports inference graph for the model specified in the pipeline config.
    Args:
      input_type: Type of input for the graph. Can be one of [`image_tensor`,
        `tf_example`].
      pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
      trained_checkpoint_prefix: Path to the trained checkpoint file.
      output_directory: Path to write outputs.
      optimize_graph: Whether to optimize graph using Grappler.
      output_collection_name: Name of collection to add output tensors to.
        If None, does not add output tensors to a collection.
    """
    detection_model = model_builder.build(pipeline_config.model,
                                          is_training=False)
    _export_inference_graph(input_type, detection_model,
                            pipeline_config.eval_config.use_moving_averages,
                            trained_checkpoint_prefix, output_directory,
                            optimize_graph, output_collection_name)


def main(_):
    assert FLAGS.pipeline_config_path, '`pipeline_config_path` is missing'
    assert FLAGS.trained_checkpoint_prefix, (
           '`trained_checkpoint_prefix` is missing')
    assert FLAGS.output_directory, '`output_directory` is missing'
    assert FLAGS.version, '`version` number is missing'
    assert FLAGS.model_name, '`model_name` is missing'

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    export_inference_graph(
        FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_prefix,
        FLAGS.output_directory)


if __name__ == '__main__':
    tf.app.run()