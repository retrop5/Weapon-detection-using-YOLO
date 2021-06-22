import cv2
import numpy as np


net = cv2.dnn.readNet("yolov3_training.weights", "yolov3_testing.cfg")
classes = ["Weapon"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))          #idu bounding box ge random color kodakke


def value():
    val = input("Enter file name or press enter to start webcam : \n") #output console ali ond print statement mulaka webcam athava existing data use madakke
    if val == "":
        val = 0
    return val


cap = cv2.VideoCapture(value()) #video thogolokke

while True:
    _, img = cap.read()
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) #4d blob create maadute, image resize maadi, sclae factor kododu amele red blue filter swap maadodu

    net.setInput(blob)
    outs = net.forward(output_layers)   #op inda predict thogondi netowrk wide inference run maadodu

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:    #prathi grid BB ge weak detection i.e. iou<0.5 na eliminate maadute
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)    #nmsuppression haakodu
    print(indexes)
    if indexes == 0: print("weapon detected in frame")  #console ali output kododu
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:    #mikirodge after nms BB haaki
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]    #class name jothe BB haakodu function
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)    #haaki BB with random color
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
