import cv2 as cv
import numpy as np
import os.path
import time



#Write down conf, nms thresholds,inp width/height
confThreshold = 0.20
#객체탐지 정확도 최저값
nmsThreshold = 0.40
#multiple boxes 값 조정 (한 객체 여러 레이어 방지) --detect 감소-- 0~1 --detect 증가--
inpWidth = 320
inpHeight = 320
# --faster-- 320 // 608 --more accurate--

#Load names of classes and turn that into a list
classesFile = "obj.names"
classes = None

with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#Model configuration
modelConf = 'Closet.cfg'
modelWeights = 'Closet_last.weights'


def take_pic(classID):
    class_num = 1
    while os.path.isfile('{}_{}.png'.format(classes[classID],class_num)):
        class_num+=1
    cv.imwrite('{}_{}.png'.format(classes[classID],class_num),frame, params=[cv.IMWRITE_PNG_COMPRESSION,0])
    

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                #confidences = 정확도값
                boxes.append([left, top, width, height])

                take_pic(classID)
                

                
    indices = cv.dnn.NMSBoxes (boxes,confidences, confThreshold, nmsThreshold )

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)
        # Determine size of bounging box

def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #A fancier display of the label from learnopencv.com 
    # Display the label at the top of the bounding box
    #labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #top = max(top, labelSize[1])
    #cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 #(255, 255, 255), cv.FILLED)
    # cv.rectangle(frame, (left,top),(right,bottom), (255,255,255), 1 )
    #cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    cv.putText(frame, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
   
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#Set up the net

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
#CPU 를 OPENCL로 바꾸면 GPU기반에서 실행


#Process inputs
winName = 'Smart Closet!'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 800,600)
#윈도우 사이즈 크기변경 





cap = cv.VideoCapture(0)
#lsussb -v로 인식된 usb 장치 확인 후 device번호 입력
f=time.time()
while cv.waitKey(1) < 0:

    #get frame from video
    hasFrame, frame = cap.read()

    #Create a 4D blob from a frame
    
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

    #Set the input the the net
    net.setInput(blob)
    outs = net.forward (getOutputsNames(net))

    postprocess (frame, outs)
    #show the image
    cv.imshow(winName, frame)

















