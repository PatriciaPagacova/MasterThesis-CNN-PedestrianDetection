import sys
import numpy as np
import tensorflow as tf
import cv2
import os

from utils import label_map_util as lmUtils
from utils import visualization_utils as visUtils

pictureFile = 'test2.png'
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

picture = cv2.imread(os.path.join(os.getcwd(), pictureFile))
expandedImage = np.expand_dims(picture, axis=0)
(boxes, scores, classes, num) = sess.run([detectionBoxes, detectionScores, detectionClasses, numDetections], feed_dict={imageTensor: expandedImage})
visUtils.visualize_boxes_and_labels_on_image_array(picture, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), categoryIndex,
	use_normalized_coordinates=True, line_thickness=5, min_score_thresh=treshold, skip_labels=True)
cv2.imshow('Pedestriants detection', picture)
cv2.imwrite(os.path.join(os.getcwd(), pictureFile.replace(".jpg","_detected.jpg").replace(".png","_detected.png").replace(".jpeg","_detected.jpeg")), picture)
cv2.waitKey(0)
cv2.destroyAllWindows()
