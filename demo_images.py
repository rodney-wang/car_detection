import os
import time
import numpy as np
import cv2 as cv
import cfg


def load_detect_model():
    net = cv.dnn.readNetFromDarknet(cfg.model_config_file, cfg.model_file)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def postprocess(image_array, outs, conf_thresh, nms_thresh, class_id_need=cfg.class_id_need):
    image_height = image_array.shape[0]
    image_width = image_array.shape[1]

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_thresh:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                width = int(detection[2] * image_width)
                height = int(detection[3] * image_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                left = max(0, left)
                top = max(0, top)
                width = max(0, width)
                height = max(0, height)
                if left + width > image_width:
                    width = image_width -left
                if top + height > image_height:
                    height = image_height - top
                    
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
#     print(classIds, confidences, boxes)
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    
    confidences_need = []
    boxes_need = []
    for i in indices:
        i = i[0]
        if classIds[i] in class_id_need:
            confidences_need.append(confidences[i])
            boxes_need.append(boxes[i])
    return confidences_need, boxes_need
    

def detect_obj(detect_model, image_array):
    image_height = image_array.shape[0]
    image_width = image_array.shape[1]

    image_blob = cv.dnn.blobFromImage(image_array, 1.0/255, cfg.obj_detect_input_shape, [0, 0, 0], 1, False)

    # net forward
    detect_model.setInput(image_blob)
    detections = detect_model.forward(getOutputsNames(detect_model))
    # print(detections)

    # decode result
    confidences_need, boxes_need = postprocess(image_array, detections, cfg.conf_thresh, cfg.nms_thresh)
#     print(confidences_need, boxes_need)

    detect_result = {"obj_num": 0, "obj_list": []}
    for i in range(len(confidences_need)):
        confidence = confidences_need[i]
        bbox = boxes_need[i]
        obj = {"location": {"left": bbox[0], "top": bbox[1], "width": bbox[2], "height": bbox[3]}, 
                   "obj_probability": round(confidence, 2)}
        detect_result["obj_list"].append(obj)
    detect_result["obj_num"] = len(detect_result["obj_list"])

    return detect_result

def plot_obj(img_file_path, detect_result):
    image_array = cv.imread(img_file_path)
    obj_list = detect_result['data']['obj_list']
    for obj in obj_list:
        obj_location = obj['location']
        x1 = obj_location['left']
        y1 = obj_location['top']
        x2 = x1 + obj_location['width']
        y2 = y1 + obj_location['height']
        cv.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), int(round(image_array.shape[0]/150)), 8)
    # save obj plot result    
    img_name = img_file_path.split('/')[-1]
    cv.imwrite(os.path.join('./demo_images_result/', img_name), image_array) 
    
    image_array_plot = cv.cvtColor(image_array, cv.COLOR_BGR2RGB)
    
    return image_array_plot

def detect_image(img_file_path, detect_model):
    result_need = {"code": 0, "msg": "unsuccess"}
    result_need["data"] = {}

    image_array = cv.imread(img_file_path)
    if np.shape(image_array) != ():
        print('read an image, image shape is ', image_array.shape)
    else:
        return result_need
   
    try:
        detect_result = detect_obj(detect_model, image_array)
        result_need["data"] = detect_result
        result_need["code"] = 1
        result_need["msg"] = "success"
    except Exception as e:
        print("Exception happend: \n" + str(e))
    return result_need
