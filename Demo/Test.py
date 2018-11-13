import functools
import tensorflow as tf
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.builders.image_resizer_builder import image_resizer_pb2
import object_detection.builders.image_resizer_builder as image_resizer_builder
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.builders import post_processing_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import box_predictor_builder
from object_detection.core import target_assigner
from object_detection.core import post_processing
from object_detection.core import losses
from object_detection.utils import ops
from object_detection.protos import hyperparams_pb2
from object_detection.protos import post_processing_pb2
from object_detection.protos import box_predictor_pb2
from google.protobuf import text_format
import object_detection.anchor_generators.grid_anchor_generator as grid_anchor_generator
from object_detection.utils import ops as utils_ops
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
from utils import label_map_util

from utils import visualization_utils as vis_util

import Demo.faster_rcnn_resnet_v1_feature_extractor as faster_rcnn_resnet_v1_feature_extractor


def get_second_stage_box_predictor(num_classes, is_training,
                                   predict_masks, masks_are_class_agnostic):
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(get_second_stage_box_predictor_text_proto(), box_predictor_proto)
    if predict_masks:
        text_format.Merge(add_mask_to_second_stage_box_predictor_text_proto(masks_are_class_agnostic),
                          box_predictor_proto)

    return box_predictor_builder.build(
        hyperparams_builder.build,
        box_predictor_proto,
        num_classes=num_classes,
        is_training=is_training)


def get_second_stage_box_predictor_text_proto():
    box_predictor_text_proto = """
      mask_rcnn_box_predictor {
        fc_hyperparams {
          op: FC
          activation: NONE
          regularizer {
            l2_regularizer {
              weight: 0.0005
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
        use_dropout: false
        dropout_keep_probability: 1.0
      }
    """
    return box_predictor_text_proto


def add_mask_to_second_stage_box_predictor_text_proto(masks_are_class_agnostic=False):
    agnostic = 'true' if masks_are_class_agnostic else 'false'
    box_predictor_text_proto = """
      mask_rcnn_box_predictor {
        predict_instance_masks: true
        masks_are_class_agnostic: """ + agnostic + """
        mask_height: 14
        mask_width: 14
        conv_hyperparams {
          op: CONV
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            truncated_normal_initializer {
              stddev: 0.01
            }
          }
        }
      }
    """
    return box_predictor_text_proto


def build_arg_scope_with_hyperparams(hyperparams_text_proto, is_training):
    hyperparams = hyperparams_pb2.Hyperparams()
    text_format.Merge(hyperparams_text_proto, hyperparams)
    return hyperparams_builder.build(hyperparams, is_training=is_training)


def build_model(is_training=False, use_matmul_crop_and_resize=False, softmax_second_stage_classification_loss=True,
                hard_mining=False, predict_masks=False, masks_are_class_agnostic=False):
    faeture_extractor = faster_rcnn_resnet_v1_feature_extractor.FasterRCNNResnet50FeatureExtractor(is_training,
                                                                                                   first_stage_features_stride=16)
    anchor_scales = (0.25, 0.5, 1.0, 2.0)
    anchor_aspect_ratios = (0.5, 1.0, 2.0)
    anchor_stride = (16, 16)
    first_stage_anchor_generator = grid_anchor_generator.GridAnchorGenerator(scales=anchor_scales,
                                                                             aspect_ratios=anchor_aspect_ratios,
                                                                             anchor_stride=anchor_stride)

    image_resizer_text_proto = """
          keep_aspect_ratio_resizer {
            min_dimension: 600
            max_dimension: 1024
          }
        """
    image_resizer_config = image_resizer_pb2.ImageResizer()
    text_format.Merge(image_resizer_text_proto, image_resizer_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)

    first_stage_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN',
        'proposal',
        use_matmul_gather=False)

    first_stage_box_predictor_hyperparams_text_proto = """
          op: CONV
          activation: RELU
          regularizer {
            l2_regularizer {
              weight: 0.00004
            }
          }
          initializer {
            truncated_normal_initializer {
              stddev: 0.03
            }
          }
        """
    first_stage_box_predictor_arg_scope_fn = (
        build_arg_scope_with_hyperparams(
            first_stage_box_predictor_hyperparams_text_proto, is_training))

    first_stage_sampler = sampler.BalancedPositiveNegativeSampler(
        positive_fraction=0.5, is_static=False)

    first_stage_non_max_suppression_fn = functools.partial(
        post_processing.batch_multiclass_non_max_suppression,
        score_thresh=0.0,
        iou_thresh=0.699999988079,
        max_size_per_class=100,
        max_total_size=100,
        use_static_shapes=False)

    crop_and_resize_fn = (
        ops.matmul_crop_and_resize
        if use_matmul_crop_and_resize else ops.native_crop_and_resize)

    second_stage_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN', 'detection',
        use_matmul_gather=False)

    second_stage_sampler = sampler.BalancedPositiveNegativeSampler(
        positive_fraction=1.0, is_static=False)

    post_processing_text_proto = """
          batch_non_max_suppression {
            score_threshold: 0.300000011921
            iou_threshold: 0.500000023842
            max_detections_per_class: 100
            max_total_detections: 100
            use_static_shapes: """ + '{}'.format(False) + """
          }
        """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    second_stage_non_max_suppression_fn, _ = post_processing_builder.build(post_processing_config)

    if softmax_second_stage_classification_loss:
        second_stage_classification_loss = (
            losses.WeightedSoftmaxClassificationLoss())
    else:
        second_stage_classification_loss = (
            losses.WeightedSigmoidClassificationLoss())

    hard_example_miner = None
    if hard_mining:  # ???
        hard_example_miner = losses.HardExampleMiner(
            num_hard_examples=1,
            iou_threshold=0.99,
            loss_type='both',
            cls_loss_weight=1.0,
            loc_loss_weight=2.0,
            max_negatives_per_positive=None)

    common_kwargs = {
        'is_training': is_training,
        'num_classes': 90,
        'image_resizer_fn': image_resizer_fn,
        'feature_extractor': faeture_extractor,
        'number_of_stages': 2,
        'first_stage_anchor_generator': first_stage_anchor_generator,
        'first_stage_target_assigner': first_stage_target_assigner,
        'first_stage_atrous_rate': 1,
        'first_stage_box_predictor_arg_scope_fn': first_stage_box_predictor_arg_scope_fn,
        'first_stage_box_predictor_kernel_size': 3,
        'first_stage_box_predictor_depth': 512,
        'first_stage_minibatch_size': 256,
        'first_stage_sampler': first_stage_sampler,
        'first_stage_non_max_suppression_fn':
            first_stage_non_max_suppression_fn,
        'first_stage_max_proposals': 100,
        'first_stage_localization_loss_weight': 2.0,
        'first_stage_objectness_loss_weight': 1.0,
        'crop_and_resize_fn': crop_and_resize_fn,
        'initial_crop_size': 14,
        'maxpool_kernel_size': 2,
        'maxpool_stride': 2,
        'second_stage_target_assigner': second_stage_target_assigner,
        'second_stage_mask_rcnn_box_predictor': get_second_stage_box_predictor(
            num_classes=90,
            is_training=is_training,
            predict_masks=predict_masks,
            masks_are_class_agnostic=masks_are_class_agnostic),
        'second_stage_batch_size': 256,
        'second_stage_sampler': second_stage_sampler,
        'second_stage_non_max_suppression_fn': second_stage_non_max_suppression_fn,
        'second_stage_score_conversion_fn': tf.nn.softmax,
        'second_stage_localization_loss_weight': 2.0,
        'second_stage_classification_loss_weight': 1.0,
        'second_stage_classification_loss': second_stage_classification_loss,
        'second_stage_mask_prediction_loss_weight': 1.0,
        'hard_example_miner': hard_example_miner,
        'parallel_iterations': 16,
        'add_summaries': True,
        'clip_anchors_to_image': False,
        'use_static_shapes': False,
        'resize_masks': True,
    }

    faster_rcnn = faster_rcnn_meta_arch.FasterRCNNMetaArch(**common_kwargs)
    return faster_rcnn


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
    PATH_TO_LABELS = os.path.join(PATH, 'data', 'mscoco_label_map.pbtxt')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = 'D:\Images'
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)

    graph = tf.Graph()
    with graph.as_default():
        model = build_model()
        with tf.variable_scope('placeholder'):
            image_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='images')
        preprocessed_inputs, true_image_shapes = model.preprocess(image_tensor)
        prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
        output_dict = model.postprocess(prediction_dict, true_image_shapes)

        saver = tf.train.Saver()
        save_path = 'E:\\CODE\\Python\\TensorFlowTest\\Demo\\faster_rcnn_resnet50_coco_2018_01_28\\faster_rcnn_resnet50_coco_2018_01_28\\model.ckpt'

        with tf.Session() as sess:
            saver.restore(sess, save_path)
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                output_dict_ = sess.run(output_dict, feed_dict={image_tensor: image_np_expanded})
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
            plt.show()

        pass
