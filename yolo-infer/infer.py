import codecs
import paddle.fluid as fluid
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import numpy as np
import time

ues_tiny = train_parameters['use_tiny']
yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']

target_size = yolo_config['input_size']
anchors = yolo_config['anchors']
anchor_mask = yolo_config['anchor_mask']

nms_threshold = 0.4
valid_thresh = 0.4
confs_threshold = 0.5
label_dict = {}
with codecs.open('data/data7085/label_list.txt') as f:
    for line in f:
        parts = line.strip().split()
        label_dict[str(float(parts[0]))] = parts[1]
print(label_dict)
class_dim = len(label_dict)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
path = "./freeze_yolov3_model"
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)


def get_yolo_anchors_classes(class_num, anchors, anchor_mask):
    yolo_anchors = []
    yolo_classes = []
    for mask_pair in anchor_mask:
        mask_anchors = []
        for mask in mask_pair:
            mask_anchors.append(anchors[2 * mask])
            mask_anchors.append(anchors[2 * mask + 1])
        yolo_anchors.append(mask_anchors)
        yolo_classes.append(class_num)
    return yolo_anchors, yolo_classes


def draw_bbox_image(img, boxes, labels, save_name):
    """
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    # font = ImageFont.truetype("font.ttf", 25)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red')
        draw.text((xmin, ymin), label_dict[str(label)], (255, 255, 0))
    img.save(save_name)
    return img


def clip_bbox(bbox):
    """
    截断矩形框
    :param bbox:
    :return:
    """
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.ANTIALIAS)
    return img


def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = Image.open(img_path)
    img = resize_img(origin, target_size)
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img


def sigmoid(x):
    """Perform sigmoid to input numpy array"""
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def box_xywh_to_xyxy(box):
    """
    bbox 两种形式的转换，左上角和宽高---->左上角|右下角
    :param box:
    :return:
    """
    shape = box.shape
    assert shape[-1] == 4, "Box shape[-1] should be 4."

    box = box.reshape((-1, 4))
    box[:, 0], box[:, 2] = box[:, 0] - box[:, 2] / 2, box[:, 0] + box[:, 2] / 2
    box[:, 1], box[:, 3] = box[:, 1] - box[:, 3] / 2, box[:, 1] + box[:, 3] / 2
    box = box.reshape(shape)
    return box


def box_iou_xyxy(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)


def rescale_box_in_input_image(boxes, im_shape, input_size):
    """Scale (x1, x2, y1, y2) box of yolo output to input image"""
    h, w = im_shape
    fx = w / input_size
    fy = h / input_size
    boxes[:, 0] *= fx
    boxes[:, 1] *= fy
    boxes[:, 2] *= fx
    boxes[:, 3] *= fy
    boxes[boxes < 0] = 0
    boxes[:, 2][boxes[:, 2] > (w - 1)] = w - 1
    boxes[:, 3][boxes[:, 3] > (h - 1)] = h - 1
    return boxes


def get_yolo_detection(preds, anchors, class_num, img_height, img_width):
    """Get yolo box, confidence score, class label from Darknet53 output"""
    preds_n = np.array(preds)
    n, c, h, w = preds_n.shape
    print(preds_n.shape, anchors)
    anchor_num = len(anchors) // 2
    preds_n = preds_n.reshape([n, anchor_num, class_num + 5, h, w]).transpose((0, 1, 3, 4, 2))
    preds_n[:, :, :, :, :2] = sigmoid(preds_n[:, :, :, :, :2])
    preds_n[:, :, :, :, 4:] = sigmoid(preds_n[:, :, :, :, 4:])

    pred_boxes = preds_n[:, :, :, :, :4]
    pred_confs = preds_n[:, :, :, :, 4]
    pred_scores = preds_n[:, :, :, :, 5:] * np.expand_dims(pred_confs, axis=4)

    grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1))
    grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w))
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors_s = np.array([(an_w, an_h) for an_w, an_h in anchors])
    anchor_w = anchors_s[:, 0:1].reshape((1, anchor_num, 1, 1))
    anchor_h = anchors_s[:, 1:2].reshape((1, anchor_num, 1, 1))

    pred_boxes[:, :, :, :, 0] += grid_x
    pred_boxes[:, :, :, :, 1] += grid_y
    pred_boxes[:, :, :, :, 2] = np.exp(pred_boxes[:, :, :, :, 2]) * anchor_w
    pred_boxes[:, :, :, :, 3] = np.exp(pred_boxes[:, :, :, :, 3]) * anchor_h

    pred_boxes[:, :, :, :, 0] = pred_boxes[:, :, :, :, 0] * img_width / w
    pred_boxes[:, :, :, :, 1] = pred_boxes[:, :, :, :, 1] * img_height / h
    pred_boxes[:, :, :, :, 2] = pred_boxes[:, :, :, :, 2]
    pred_boxes[:, :, :, :, 3] = pred_boxes[:, :, :, :, 3]

    pred_boxes = box_xywh_to_xyxy(pred_boxes)
    pred_boxes = np.tile(np.expand_dims(pred_boxes, axis=4), (1, 1, 1, 1, class_num, 1))
    pred_labels = np.zeros_like(pred_scores) + np.arange(class_num)

    return pred_boxes.reshape((n, -1, 4)), pred_scores.reshape((n, -1)), pred_labels.reshape((n, -1))


def get_all_yolo_pred(outputs, yolo_anchors, yolo_classes, input_shape):
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    for output, anchors, classes in zip(outputs, yolo_anchors, yolo_classes):
        pred_boxes, pred_scores, pred_labels = get_yolo_detection(output, anchors, classes, input_shape[0],
                                                                  input_shape[1])
        all_pred_boxes.append(pred_boxes)
        all_pred_labels.append(pred_labels)
        all_pred_scores.append(pred_scores)
    pred_boxes = np.concatenate(all_pred_boxes, axis=1)
    pred_scores = np.concatenate(all_pred_scores, axis=1)
    pred_labels = np.concatenate(all_pred_labels, axis=1)

    return pred_boxes, pred_scores, pred_labels


def calc_nms_box(pred_boxes, pred_scores, pred_labels, valid_thresh=0.4, nms_thresh=0.45, nms_topk=400):
    output_boxes = np.empty((0, 4))
    output_scores = np.empty(0)
    output_labels = np.empty(0)
    for boxes, labels, scores in zip(pred_boxes, pred_labels, pred_scores):
        valid_mask = scores > valid_thresh
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        labels = labels[valid_mask]

        score_sort_index = np.argsort(scores)[::-1]
        boxes = boxes[score_sort_index][:nms_topk]
        scores = scores[score_sort_index][:nms_topk]
        labels = labels[score_sort_index][:nms_topk]

        for c in np.unique(labels):
            c_mask = labels == c
            c_boxes = boxes[c_mask]
            c_scores = scores[c_mask]

            detect_boxes = []
            detect_scores = []
            detect_labels = []
            while c_boxes.shape[0]:
                detect_boxes.append(c_boxes[0])
                detect_scores.append(c_scores[0])
                detect_labels.append(c)
                if c_boxes.shape[0] == 1:
                    break
                iou = box_iou_xyxy(detect_boxes[-1].reshape((1, 4)), c_boxes[1:])
                c_boxes = c_boxes[1:][iou < nms_thresh]
                c_scores = c_scores[1:][iou < nms_thresh]

            output_boxes = np.append(output_boxes, detect_boxes, axis=0)
            output_scores = np.append(output_scores, detect_scores)
            output_labels = np.append(output_labels, detect_labels)
    return output_boxes, output_scores, output_labels


def infer(image_path):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    origin, tensor_img, resized_img = read_image(image_path)
    t1 = time.time()
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img},
                            fetch_list=fetch_targets)
    period = time.time() - t1
    print("predict cost time:{0}".format("%2.2f sec" % period))
    input_w, input_h = origin.size[0], origin.size[1]
    yolo_anchors, yolo_classes = get_yolo_anchors_classes(class_dim, anchors, anchor_mask)
    pred_boxes, pred_scores, pred_labels = get_all_yolo_pred(batch_outputs, yolo_anchors, yolo_classes,
                                                             (target_size[1], target_size[2]))
    boxes, scores, labels = calc_nms_box(pred_boxes, pred_scores, pred_labels, valid_thresh, nms_threshold)
    boxes = rescale_box_in_input_image(boxes, [input_h, input_w], target_size[1])
    print("result boxes: ", boxes)
    print("result scores:", scores)
    print("result labels:", labels)
    last_dot_index = image_path.rfind('.')
    out_path = image_path[:last_dot_index]
    out_path += '-reslut.jpg'
    draw_bbox_image(origin, boxes, labels, out_path)


if __name__ == '__main__':
    # image_path = sys.argv[1]
    name_list = os.listdir('data/data7085/test/images/')
    for name_ in name_list:
        image_path = os.path.join('data/data7085/test/images', name_)
        infer(image_path)