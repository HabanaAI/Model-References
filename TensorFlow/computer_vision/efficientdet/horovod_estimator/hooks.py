import io
import itertools
import os
import time

import cv2

from TensorFlow.common.horovod_helpers import hvd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
from PIL import ImageDraw, ImageFont
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from collections import OrderedDict

from horovod_estimator.utis import hvd_info_rank0, is_rank0

np.seterr(divide='ignore', invalid='ignore')


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    """
    SessionRunHook that will broadcast all global variables from root rank
    to all other processes during initialization.

    This is necessary to ensure consistent initialization of all workers when
    training is started with random weights or restored from a checkpoint.
    """

    def __init__(self, root_rank, pretrained_model_path=None, exclusions=[], device='/device:HPU:0', model_dir=None):
        """Construct a new BroadcastGlobalVariablesHook that will broadcast all
        global variables from root rank to all other processes during initialization.

        Args:
          root_rank:
            Rank that will send data, other ranks will receive data.
          device:
            Device to be used for broadcasting. Uses GPU by default
            if Horovod was build with HOROVOD_GPU_BROADCAST.
        """
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.root_rank = root_rank
        self.bcast_op = None
        self.device = device
        self._pretrained_model_path = pretrained_model_path
        self._saver = None
        self._exclusions = set(exclusions)
        self._variables_to_restore = []
        self._model_dir = model_dir

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                self.bcast_op = hvd.broadcast_global_variables(self.root_rank)

        if self._model_dir is not None:
            checkpoint_path = checkpoint_management.latest_checkpoint(self._model_dir)
            if checkpoint_path is not None and not checkpoint_path.endswith('model.ckpt-0'):
                hvd_info_rank0('>>>>> model_dir {} has checkpoint {}, not using pretrained_model_path <<<<<'.
                               format(self._model_dir, checkpoint_path))
                return

        if self._pretrained_model_path is not None and len(self._pretrained_model_path) > 0 and is_rank0():
            reader = pywrap_tensorflow.NewCheckpointReader(self._pretrained_model_path)
            var_to_shape_map = sorted(reader.get_variable_to_shape_map())

            self._exclusions.add('global_step')

            for var in tf.global_variables():
                if var.op.name in var_to_shape_map:
                    excluded = False
                    for exclusion in self._exclusions:
                        if var.op.name.startswith(exclusion):
                            excluded = True
                            break
                    if not excluded:
                        self._variables_to_restore.append(var)

            self._saver = tf.train.Saver(var_list=self._variables_to_restore)

    def after_create_session(self, session, coord):
        if self._saver:
            hvd_info_rank0('>>>>> begin to load weights from {}, restore variables length {}, without variables {}'
                           .format(self._pretrained_model_path, len(self._variables_to_restore), self._exclusions))
            self._saver.restore(session, self._pretrained_model_path)
            hvd_info_rank0('<<<<< end to load weights')

        hvd_info_rank0('>>>>> broadcast global variables begin during after_create_session')
        session.run(self.bcast_op)
        hvd_info_rank0('<<<<< broadcast global variables end during after_create_session ')


class LoggingTensorHook(tf.train.SessionRunHook):
    def __init__(self, named_tensor, summary_dir=None, every_n_iter=100, use_all_reduce=False):
        super(LoggingTensorHook, self).__init__()
        self._named_tensor = named_tensor
        self._every_n_iter = every_n_iter
        self._summary_dir = summary_dir
        self._step = 0
        self._use_all_reduce = use_all_reduce

        self._tic = time.time()
        self._avg_ops = {}
        self._global_step_tensor = None

    def begin(self):
        if self._use_all_reduce:
            self._avg_ops = OrderedDict({'{}'.format(tag): hvd.allreduce(basic_session_run_hooks._as_graph_element(tensor))
                                         for (tag, tensor) in self._named_tensor.items()})
        else:
            self._avg_ops = OrderedDict({'{}'.format(tag): basic_session_run_hooks._as_graph_element(tensor)
                                         for (tag, tensor) in self._named_tensor.items()})

        self._global_step_tensor = tf.train.get_or_create_global_step()
        self._avg_ops['step'] = self._global_step_tensor

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._tic = time.time()
        if self._step % self._every_n_iter == 0:
            return SessionRunArgs(fetches=self._avg_ops)


    def _log_tensors(self, tensor_values):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)

        stats = []
        for tag, tensor in tensor_values.items():
            stats.append('%s = %s' % (tag, tensor))

        stats.append('%s = %s' % ('step_time', time.time() - self._tic))

        if self._use_all_reduce:
            logging_head = 'logging all reduce tensors'
        else:
            logging_head = 'logging tensors'

        hvd_info_rank0("{}: {}".format(logging_head, ", ".join(stats)))
        np.set_printoptions(**original)

    def _summary(self, tensor_values):
        if self._summary_dir:
            writer = tf.summary.FileWriterCache.get(self._summary_dir)
            this_summary = tf.Summary()
            for tag, value in tensor_values.items():
                this_summary.value.add(tag=tag, simple_value=value)
                writer.add_summary(this_summary, tensor_values['step'])

            writer.flush()

    def after_run(self, run_context, run_values):
        if self._step % self._every_n_iter == 0:
            if is_rank0() or not self._use_all_reduce:
                avg_values = run_values.results
                self._log_tensors(avg_values)
                self._summary(avg_values)

        self._step += 1


def make_image(tensor):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)


def to_fix_format(i):
    if isinstance(i, int) or isinstance(i, np.int32) or isinstance(i, np.int64):
        return str(i)
    else:
        return '{:.2f}'.format(i)


def draw_text_image(text, size=(224, 224)):
    # make a blank image for the text, initialized to transparent text color
    img = Image.new('RGB', size, (0, 0, 0))
    d = ImageDraw.Draw(img)

    # to-do: how to align
    # if len(text) <= 2:
    #     font_size = 100
    #     xy = (80, 60)
    # else:
    #     font_size = 40
    #     xy = (60, 90)

    xy = (10, 10)
    font_size = 20

    # get a font
    fnt = ImageFont.truetype('assets/fonts/FreeMono.ttf', size=font_size)
    # get a drawing context
    d.text(xy, text, font=fnt, fill=(255, 255, 255))
    return img


def scale_to_uint8(features_tensor):
    if len(features_tensor) > 0:
        min_f = np.min(features_tensor)
        max_f = np.max(features_tensor)
        features_tensor = (features_tensor - min_f) / (max_f - min_f) * 255
        features_tensor = features_tensor.astype(np.uint8)
    return features_tensor


def top_k_text(prob_array, k):
    sort_idx = np.argsort(prob_array)[-k:][::-1]
    top_k_prob = prob_array[sort_idx]
    top_k_idx = sort_idx

    result = ''
    for i in range(k):
        result += '{}: {}\n'.format(top_k_idx[i], to_fix_format(top_k_prob[i]))

    return result.strip()


def find_xy(img, threshold=0.8, percentile=False):
    x_offset = 3
    y_offset = 3
    img = img[x_offset: -x_offset, y_offset: -y_offset]
    threshold = threshold * np.max(img)
    idx = np.argwhere(img > threshold)
    x_min = np.min(idx[:, 1]) + x_offset
    x_max = np.max(idx[:, 1]) + x_offset

    y_min = np.min(idx[:, 0]) + y_offset
    y_max = np.max(idx[:, 0]) + y_offset

    if percentile:
        h, w = img.shape
        x_min = x_min / w
        x_max = x_max / w
        y_min = y_min / h
        y_max = y_max / h

        x_max = min(1.0, x_max)
        y_max = min(1.0, y_max)

    return x_min, y_min, x_max, y_max


def draw_box(img_tensor, box):
    img = Image.fromarray(img_tensor)
    d = ImageDraw.Draw(img)
    x_min, y_min, x_max, y_max = box
    d.rectangle(((x_min, y_min), (x_max, y_max)), fill='white', outline=3)
    return np.asarray(img, dtype=np.uint8)


def show_images(filenames, images, raw_images, heat_map_features, labels, probs, global_step, max_images,
                summary_writer, prefix='train'):
    if summary_writer is not None:
        assert images is not None and labels is not None and probs is not None
        n, height, width, channel = images.shape
        padding_255 = np.ones([height, 1, channel], dtype=np.uint8) * 255
        padding_1 = np.ones([height, 1, channel], dtype=np.float32)

        filenames_tensor_list = []
        raw_images_tensor_list = []
        images_tensor_list = []
        heat_map_tensor_list = []
        label_tensor_list = []
        max_images = min(max_images, n)

        for i in range(max_images):
            images_tensor_list.append(images[i])
            images_tensor_list.append(padding_1)

            if raw_images is not None:
                raw_images_tensor_list.append(raw_images[i])
                raw_images_tensor_list.append(padding_1)

            if heat_map_features is not None:
                cam = heat_map_features[i][:, :, labels[i]]
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                heat_map = cv2.applyColorMap(cv2.resize(cam_img, (height, width)), cv2.COLORMAP_JET)
                blue_map = heat_map[:, :, -1]
                box = find_xy(blue_map)
                heat_map = draw_box(heat_map, box)
                heat_map = heat_map / 255.0
                heat_img = heat_map * 0.7 + images[i] * 0.3

                heat_map_tensor_list.append(heat_img)
                heat_map_tensor_list.append(padding_1)

            # labes & predicts
            probs_show_num = min(5, probs.shape[-1])
            text = '{}: {}\n'.format(to_fix_format(labels[i]), to_fix_format(probs[i][labels[i]])) \
                   + top_k_text(probs[i], probs_show_num)
            label_image = draw_text_image(text, (height, width))
            label_tensor_list.append(np.asarray(label_image, dtype=np.uint8))
            label_tensor_list.append(padding_255)

            filename = filenames[i]
            if isinstance(filename, bytes):
                filename = filename.decode('utf-8')

            filename = filename.split('/')[-1]
            filename_image = draw_text_image(filename, (height, width))
            filenames_tensor_list.append(np.asarray(filename_image, dtype=np.uint8))
            filenames_tensor_list.append(padding_255)

        # scale float32 to unit8
        all_tensor_list = [scale_to_uint8(np.concatenate(filenames_tensor_list, axis=1))]
        if raw_images is not None:
            all_tensor_list.append(scale_to_uint8(np.concatenate(raw_images_tensor_list, axis=1)))

        all_tensor_list.append(scale_to_uint8(np.concatenate(images_tensor_list, axis=1)))

        if heat_map_features is not None:
            all_tensor_list.append(scale_to_uint8(np.concatenate(heat_map_tensor_list, axis=1)))

        all_tensor_list.append(np.concatenate(label_tensor_list, axis=1))

        feature_heatmap_label_tensor = np.concatenate(all_tensor_list, axis=0)
        summary = Summary(value=[Summary.Value(tag='{}/features_heatmap_labels'.format(prefix),
                                               image=make_image(feature_heatmap_label_tensor))])

        summary_writer.add_summary(summary, global_step)


def plt_to_image_summary(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    image = Image.open(buf).convert('RGB')
    tensor = np.asarray(image, dtype=np.uint8)
    image = make_image(tensor)
    return image


def confusion_matrix_summary(tag, cm, classes, normalize=False, recall=True, title='Confusion matrix',
                             cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        if recall:
            s = cm.sum(axis=1)[:, np.newaxis] + np.finfo(np.float32).eps
        else:
            s = cm.sum(axis=0)[:, np.newaxis] + np.finfo(np.float32).eps

        cm = cm.astype('float') / s

    plt.close('all')

    f_size = max(5, int(0.6 * len(classes)))
    plt.figure(figsize=(f_size, f_size))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    image = plt_to_image_summary(plt)
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def roc_summary(tag, y_true, y_pred, n_classes):
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from scipy import interp

    classes = [i for i in range(n_classes)]
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2

    # plt.figure(0)
    # plt.plot(fpr[2], tpr[2], color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.close('all')
    f_size = max(5, int(0.6 * len(classes)))
    plt.figure(figsize=(f_size, f_size))

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    image = Image.open(buf).convert('RGB')
    tensor = np.asarray(image, dtype=np.uint8)
    image = make_image(tensor)

    return Summary(value=[Summary.Value(tag=tag, image=image)])


class EvalImageVisualizationHook(tf.train.SessionRunHook):
    def __init__(self, images_name, labels_name, filenames_name, probs_name, raw_images_name=None,
                 heat_map_features_name=None, every_n_steps=100, summary_dir=None, max_images=8):
        self._images_name = images_name
        self._labels_name = labels_name
        self._heat_map_features_name = heat_map_features_name
        self._probs_name = probs_name
        self._every_n_steps = every_n_steps
        self._summary_dir = summary_dir
        self._step = 0
        self._run_begin = 0
        self._run_end = 0
        self._max_images = max_images
        self._duration = 0.0
        self._raw_images_name = raw_images_name
        self._filenames_name = filenames_name

    def begin(self):
        self._summary_writer = tf.summary.FileWriterCache.get(self._summary_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()

    def before_run(self, run_context):
        self._run_begin = time.time()
        if self._step > 0 and self._step % self._every_n_steps == 0:
            arg_map = {}

            for name in [self._images_name, self._labels_name, self._filenames_name, self._raw_images_name,
                         self._heat_map_features_name, self._probs_name]:
                if name is not None:
                    try:
                        arg_map[name] = basic_session_run_hooks._as_graph_element(name)
                    except Exception as e:
                        if not self.is_logged:
                            tf.logging.error('{} error {}'.format(name, e))
                            self.is_logged = True

            arg_map['global_step'] = self._global_step_tensor
            return SessionRunArgs(arg_map)

    def _log_and_record(self, step):
        if self._summary_writer is not None:
            if self._total_batch_size:
                img_per_sec_tag = 'eval/img_per_sec'
                img_per_sec_tag_value = self._total_batch_size / (self._run_end - self._run_begin)
                sec_per_img_tag = 'eval/sec_per_img'
                sec_per_img_tag_value = 1 / img_per_sec_tag_value * 1000
                summary = Summary(value=[Summary.Value(tag=img_per_sec_tag, simple_value=img_per_sec_tag_value),
                                         Summary.Value(tag=sec_per_img_tag, simple_value=sec_per_img_tag_value)])
                logging.info("%s: %g, %s: %g ms, step: %g",
                             img_per_sec_tag, img_per_sec_tag_value, sec_per_img_tag, sec_per_img_tag_value, step)
                self._summary_writer.add_summary(summary, step)

    def after_run(self, run_context, run_values):
        self._run_end = time.time()
        self._duration += self._run_end - self._run_begin
        # not use step 0 to warmup
        if self._step > 0 and self._step % self._every_n_steps == 0:
            results = run_values.results
            global_step = results['global_step']

            images = get_key_or_none(results, self._images_name)
            labels = get_key_or_none(results, self._labels_name)
            filenames = get_key_or_none(results, self._filenames_name)
            raw_images = get_key_or_none(results, self._raw_images_name)
            heat_map_features = get_key_or_none(results, self._heat_map_features_name)
            probs = get_key_or_none(results, self._probs_name)

            self._total_batch_size = len(images) * hvd.size()

            self._log_and_record(self._step + global_step)
            show_images(filenames, images, raw_images, heat_map_features, labels, probs, self._step + global_step,
                        self._max_images, self._summary_writer, prefix='eval')

        self._step += 1

    def end(self, session):
        total_image_count = self._step * self._total_batch_size
        image_per_second = total_image_count / self._duration
        second_per_image = self._duration / total_image_count * 1000
        logging.info('total {}: {}, {}: {}, {}: {}, {}: {} ms'.format('duration', self._duration, 'total_image_count',
                                                                      total_image_count, 'image_per_second',
                                                                      image_per_second, 'second_per_image',
                                                                      second_per_image))


class SpeedHook(basic_session_run_hooks.StepCounterHook):
    def __init__(self, summary_dir, batch_size, every_n_steps=100):
        super(SpeedHook, self).__init__(every_n_steps=every_n_steps, output_dir=summary_dir)
        self._total_batch_size = batch_size * hvd.size()

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        steps_per_sec = elapsed_steps / elapsed_time
        if self._summary_writer is not None:
            if self._total_batch_size:
                image_tag = 'images_sec'
                image_count = float(steps_per_sec) * self._total_batch_size
                summary = Summary(value=[Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec),
                                         Summary.Value(tag=image_tag, simple_value=image_count)])
                logging.info("%s: %g, %s: %g, step: %g", self._summary_tag, steps_per_sec, image_tag, image_count,
                             global_step)
            else:
                summary = Summary(value=[Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec)])
                logging.info("%s: %g, step: %g", self._summary_tag, steps_per_sec, global_step)

            self._summary_writer.add_summary(summary, global_step)


def get_key_or_none(d, key):
    if key in d:
        return d[key]
    else:
        return None

class PrefillStagingAreasHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        # TODO: This assumes TF collections are ordered; is this safe?
        enqueue_ops = tf.get_collection('STAGING_AREA_PUTS')
        for i in range(len(enqueue_ops)):
            session.run(enqueue_ops[:i + 1])


class OomReportingHook(tf.train.SessionRunHook):
    def before_run(self, run_context):
        return SessionRunArgs(fetches=[],  # no extra fetches
                              options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
