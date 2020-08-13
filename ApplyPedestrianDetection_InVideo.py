import sys
import numpy as np
import tensorflow as tf
import cv2
import os
import time

from utils import label_map_util as lmUtils
from utils import visualization_utils as visUtils

videoFile = 'TestovacieVideo.mp4'
treshold = 0.50

graph = tf.Graph()
with graph.as_default():
    graphDef = tf.GraphDef()
    with tf.gfile.GFile(os.path.join(os.getcwd(),'inference_graph', 'frozen_inference_graph.pb'), 'rb') as frozenInferenceGraph:
        readedGraph = frozenInferenceGraph.read()
        graphDef.ParseFromString(readedGraph)
        tf.import_graph_def(graphDef, name='')
    sess = tf.Session(graph=graph)

imageTensor = graph.get_tensor_by_name('image_tensor:0')
detectionBoxes = graph.get_tensor_by_name('detection_boxes:0')
detectionScores = graph.get_tensor_by_name('detection_scores:0')
detectionClasses = graph.get_tensor_by_name('detection_classes:0')
numDetections = graph.get_tensor_by_name('num_detections:0')

labelMap = lmUtils.load_labelmap(os.path.join(os.getcwd(),'training','labelmap.pbtxt'))
categories = lmUtils.convert_label_map_to_categories(labelMap, max_num_classes=1, use_display_name=True)
categoryIndex = lmUtils.create_category_index(categories)
video = cv2.VideoCapture(os.path.join(os.getcwd(), videoFile))
start = time.time()
countOfDetectedFrames = 0;


while(video.isOpened()):

    ret, frame = video.read()
    expandedFrame = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run([detectionBoxes, detectionScores, detectionClasses, numDetections], feed_dict={imageTensor: expandedFrame})
    visUtils.visualize_boxes_and_labels_on_image_array(frame, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), categoryIndex,
        use_normalized_coordinates=True, line_thickness=5, min_score_thresh=treshold, skip_labels=True)
    cv2.imshow('Pedestriants detection', frame)

    countOfDetectedFrames = countOfDetectedFrames + 1
    elapsedSeconds = time.time() - start
    print('Images[' + str(countOfDetectedFrames) + '] per second [' + str(elapsedSeconds) + ']  ==>  ' + str(countOfDetectedFrames / elapsedSeconds))
    if cv2.waitKey(1) == 27: break

cv2.destroyAllWindows()
video.release()
