from core.functions import *
import core.utils as utils
import numpy as np
import cv2
from collections import deque
import tensorflow as tf
import time
import colorsys
import os
from tensorflow.compat.v1 import ConfigProto
from keras.models import load_model, model_from_json
from tensorflow.python.saved_model import tag_constants



from keras.models import Sequential
from keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout, Flatten, Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)



class YOLO:
    def __init__(self, iou, score, input_size, weights) -> None:
        self.iou = iou
        self.score = score
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        saved_model_loaded = tf.saved_model.load(
            weights, tags=[tag_constants.SERVING])
        self.infer = saved_model_loaded.signatures['serving_default']
        self.input_size = input_size
        self.class_names = {0: "Person"}
        self.allowed_classes = list(self.class_names.values())
        num_classes = len(self.allowed_classes)
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.)
                      for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou,
            score_threshold=self.score
        )

        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0],
                     valid_detections.numpy()[0]]
        image = utils.draw_bbox(frame, bboxes=pred_bbox, colors=self.colors, classes=self.class_names)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


class CNN:
    def __init__(self, weights_path, image_height=128, image_width=128, sequence_length=25):
    
        self.convlstm_model= create_convlstm_model()
        self.convlstm_model.load_weights(weights_path)
       
        self.image_height, self.image_width = image_height, image_width
        self.sequence_length = sequence_length
        self.classes_list = ["drowning", "normal"]

class FrameProcessor:
    def __init__(self, yolo, cnn):
        self.yolo: YOLO = yolo
        self.cnn_model: CNN = cnn
        self.frames_queue = deque(maxlen=cnn.sequence_length)

    def process_frame(self, frame):
        frame = self.yolo.process_frame(frame)

        resized_frame = cv2.resize(
            frame, (self.cnn_model.image_height, self.cnn_model.image_width))
        normalized_frame = resized_frame / 255
        self.frames_queue.append(normalized_frame)
        if len(self.frames_queue) == self.cnn_model.sequence_length:
            predicted_labels_probabilities = self.cnn_model.convlstm_model.predict(
                np.expand_dims(self.frames_queue, axis=0), verbose=0)[0]
            predicted_label = np.argmax(predicted_labels_probabilities)

         

            predicted_class_name = self.cnn_model.classes_list[predicted_label]
            cv2.putText(frame, predicted_class_name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame


def detect_video(frame_processor, video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        (grabbed, frame) = cap.read()
        if not grabbed:
            break
        frame = frame_processor.process_frame(frame)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

def create_convlstm_model():
    # We will use a Sequential model for model construction
        model = Sequential()
        
     
 
    # Define the Model Architecture.
    ########################################################################################################################
    
        model.add(ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'relu',data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape = (25,  128, 128, 3)))
    
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.2)))
        model.add(ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = 'relu', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.2)))
        model.add(ConvLSTM2D(filters = 14, kernel_size = (3, 3), activation = 'relu', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.2)))
        model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = 'relu', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        
        model.add(TimeDistributed(Dropout(0.2)))
        model.add(Flatten()) 
        model.add(Dense(2, activation = "softmax"))
    
    ########################################################################################################################
    # Return the constructed convlstm model.
        return model

if __name__ == "__main__":
    yolo_weights = "./checkpoints/yolov4-416"
    cnn_weights = ".Ex4_3DCNN"
    

    yolo = YOLO(iou=0.45, score=0.7, input_size=416, weights=yolo_weights)
    cnn = CNN(cnn_weights)

    # for each request create new processor and call detect video with the video path
    frame_processor = FrameProcessor(yolo, cnn)
    detect_video(frame_processor, "./data/Mixed_vid.mp4")
