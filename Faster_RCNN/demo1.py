import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import lxml.etree as etree
from object_detection.utils import dataset_util

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow.contrib.slim as slim
from object_detection.legacy import trainer
from object_detection.builders import model_builder
from Faster_RCNN.Model import model_builder

sys.path.append(r'D:\Software\Miniconda\envs\tensorflow\Lib\site-packages\tensorflow\models\research')
from object_detection.utils import config_util
from object_detection.utils import ops as utils_ops

from Faster_RCNN.Tools.generate_random_box import random_rpn

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
from utils import label_map_util

from Faster_RCNN import config

from utils import visualization_utils as vis_util
slim.conv2d

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def convert(output_dict):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


if __name__ == '__main__':
    PATH = r'D:\Software\Miniconda\envs\tensorflow\Lib\site-packages\tensorflow\models\research\object_detection'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN\\DataSet', 'mot_label_map.pbtxt')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = 'F:\PostGraduate\DataSet\MOTFromWinDataSet'
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '00000{}'.format(i)) for i in range(1, 4)]

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)

    pipeline_config_path = 'E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN\\Model\\pipeline_muti.config'
    #pipeline_config_path = 'E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN\\Log_Temp\\pipeline.config'
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    graph = tf.Graph()
    with graph.as_default():
        model = model_builder.build(model_config=model_config, is_training=False)
        with tf.variable_scope('placeholder'):
            image_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='images')
            if not config.need_rpn:
                rpn_box = tf.placeholder(tf.float32, shape=(1, None, 4), name='rpn_box')
        if not config.need_rpn:
            rpn_class = tf.ones_like(rpn_box, dtype=tf.int32)
            rpn_class = rpn_class[:,:, :2]
            model.provide_rpn_box(rpn_box, rpn_class)
        preprocessed_inputs, true_image_shapes = model.preprocess(image_tensor)
        prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
        output_dict = model.postprocess(prediction_dict, true_image_shapes)

        saver = tf.train.Saver()
        summary = tf.summary.merge_all()
        save_path = 'E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN\\Log\\model.ckpt-6279'

        with tf.Session() as sess:
            saver.restore(sess, save_path)
            path = 'E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN\\Log\\Log_2'
            train_summary = tf.summary.FileWriter(path, sess.graph)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            i = 0
            for image_path in TEST_IMAGE_PATHS:

                with tf.gfile.GFile(image_path+'.xml', 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                width = int(data['size']['width'])
                height = int(data['size']['height'])
                xmin = []
                ymin = []
                xmax = []
                ymax = []
                if 'object' in data:
                    for obj in data['object']:
                        xmin.append(float(obj['bndbox']['xmin']) / width)
                        ymin.append(float(obj['bndbox']['ymin']) / height)
                        xmax.append(float(obj['bndbox']['xmax']) / width)
                        ymax.append(float(obj['bndbox']['ymax']) / height)
                rpn = random_rpn(zip(ymin, xmin, ymax, xmax))
                rpn = [[ymin, xmin, ymax, xmax] for (ymin, xmin, ymax, xmax) in zip(*rpn)]
                rpn = np.array(rpn)
                rpn = rpn.reshape([1, -1, 4])

                image = Image.open(image_path + '.jpg')
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                output_dict_,prediction_dict_ = sess.run([output_dict,prediction_dict], feed_dict={image_tensor: image_np_expanded, rpn_box: rpn},
                                        options=run_options,
                                        run_metadata=run_metadata)

                output_dict_['detection_classes'][0] += 1
                # Actual detection.
                # output_dict = run_inference_for_single_image(image_np, detection_graph)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict_['detection_boxes'][0],
                    output_dict_['detection_classes'][0].astype(np.uint8),
                    output_dict_['detection_scores'][0],
                    category_index,
                    instance_masks=output_dict_.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)
                print(output_dict_['detection_scores'])
                print(output_dict_['detection_classes'])
                i += 1
                train_summary.add_run_metadata(run_metadata, 'image%03d' % i)
                train_summary.flush()
                saver.save(sess, os.path.join(path, 'model.ckpt'))

            train_summary.close()
            plt.show()

        pass
