import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from New_yolo import YOLO
import os
from train import create_model, get_anchors, get_classes

input_shape = (416, 416)
# classes_path = "/home/yossi/Desktop/Csgo_yolo_model/Yolov3_inputs/classes.txt"
# Complete classes path.
classes_path = "/home/yossi/Desktop/Complete_yolo_inputs/Complete_yolo_classes"
anchors_path = '/home/yossi/Desktop/Csgo_yolo_model/model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
print("class_names", class_names)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
model = create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                     weights_path='/home/yossi/Desktop/Complete_yolo_inputs/Complete_final_yolo_models/Complete_model_2.h5')
model.compile(optimizer=Adam(lr=1e-3), loss={
    # use custom yolo_loss Lambda layer.
    'yolo_loss': lambda y_true, y_pred: y_pred})
print(model)
print(type(model))
# test_images_path="/home/yossi/Desktop/Csgo_yolo_model/Yolov3_inputs/Test_inputs"
# Comlete test inputs path.
test_images_path = "/home/yossi/Desktop/Complete_yolo_inputs/Must_have_samples"

X_t = []
y_t = []
files_t = []
classes_d = {"Knife": 1, "knife": 1, "blind kill": 2, "Blind kill": 2, "Grenade": 3, "grenade": 3, "no scope": 4,
             "No scope": 4, "Headshot": 5, "headshot": 5, "Fire": 6, "fire": 6, "Through smoke": 7, "through smoke": 7,  "Through wall": 8,
             "through wall": 8, "m4a4": 9, "sg 553": 9, "AK-47": 9, "ak-47": 9, "Galil AR": 9, "Rifle": 9, "p2000": 10,
             "glock-18": 10, "glock 18": 10, "tec": 10, "desert eagle": 10, "dual berettas": 10, "Gun": 10, "negev": 11,
             "Machine gun": 11, "Awp": 12, "awp": 12,"AWP":12, "ssg-08": 12, "ssg 08":12, "Sniper": 12, "p90": 13, "PP-Bizon": 13,
             "ump-45": 13, "SMG": 13, "xm1014": 14, "nova": 14, "Shotgun": 14}
# Number of actual object to be detected within each image.
num_of_objects = []
ind = 0
for subdir, dirs, files in os.walk(test_images_path):
    for img in files:
        if ind >= 5564:
            break
        ind += 1
        i = 0
        if img.endswith("jpg"):
            image_path = os.path.join(subdir, img)
            image = Image.open(image_path)
            X_t.append(image)
            files_t.append(img)
            for key in classes_d:
                if key in img:
                    i += 1
                    y_t.append(classes_d[key])
        num_of_objects.append(i)
        if i == 0:
            print(
                "Key not found-------------------------------------------------------------------------------------------------------------",
                img)

# model.evaluate(X_t,y_t)
print("Number of images", len(X_t))
print("Number of objects", len(y_t),"Real Labels",y_t)
print("Number of objects per image", num_of_objects)
print("Filenames", files_t)

yolo_detector = YOLO()
#print(yolo_detector)
#print(type(yolo_detector))
labels = []
labels_false = []
yt_false = []
for i in range(len(X_t)):
    print("i is",i)
    img = X_t[i]
    res, label_true, label_false = yolo_detector.detect_image(img)
    if len(label_true)==num_of_objects[i]:
        print(files_t[i])
        for r in label_true:
            for k in classes_d:
                if k in r:
                    labels.append(classes_d[k])

    elif len(label_true)<num_of_objects[i]:
        print("Fewer detections than objects---------------------------------------------------")
        print(files_t[i])
        filling=[0 for i in range(num_of_objects[i])]
       # correction=[0 for i in range(num_of_objects[i]-len(label_true))]
        #print("Corrections",correction)
        print(labels)
        label_num=[]
        print(files_t[i])
        labels.extend(filling)
        if label_true:
            for r in label_true:
                for k in classes_d:
                    if k in r:
                        label_num.append(classes_d[k])
            b=sum(num_of_objects[:i])
            e=sum(num_of_objects[:i+1])
            for n in label_num:
                if n in y_t[b:e]:
                    for l in range(b, e):
                        if n == y_t[l]:
                            labels[l] = n
                else:
                    label_false_new.append(n)




        #labels.extend(correction)

    else:
        print("Missdetection, too many objects found-------------------------------")
        print("number of objects",num_of_objects[i],"number of detections",len(label_true))
        correction = len(label_true)-num_of_objects[i]
        print("Corrections",correction)
        print(files_t[i])
        #label_true_new=[]
        label_false_new=[]
        b = sum(num_of_objects[:i])
        e = sum(num_of_objects[:i + 1])
        label_num=[]
        filling = [0 for i in range(num_of_objects[i])]
        labels.extend(filling)

        for r in label_true:
            for k in classes_d:
                if k in r:
                    label_num.append(classes_d[k])
        for n in label_num:
            if n in y_t[b:e]:
                for l in range(b, e):
                    if n == y_t[l]:
                        labels[l] = n
            else:
                label_false_new.append(n)
        #for l in label_num:
         #   print("l",l)
            #print("y_t[i:rng]",y_t[i:rng])
          #  if l in y_t[b:e]:
           #     label_true_new.append(l)
            #else:
             #   label_false_new.append(l)
        #labels.extend(label_true_new)
        #print("label_true_new",label_true_new)
        labels_false.extend(label_false_new)
        print("label_false_new",label_false_new)
        #print("y_t[i:i+rng]",y_t[i:i+rng])

    #res.show()

print("labels", labels, "length", len(labels))
print("y_t   ", y_t, "length", len(y_t))

y_t_addition=[0 for i in labels_false]
y_t.extend(y_t_addition)
labels.extend(labels_false)

precision, recall, fbeta_score, support = precision_recall_fscore_support(y_t, labels,
                                                                          labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                                                  13, 14], average="micro")
print("precision: ", precision, "\n", "Recall: ", recall, "\n", "Fbeta_score", fbeta_score, "\n", "Support: ", support)
print("y_t   ", y_t)
print("labels", labels)
print("labels_false", labels_false)
print("length_labels", len(labels), "length_labels_false", len(labels_false))

# y_t_updated=y_t+[0 for i in range(len(labels_false))]
# labels_updated=labels+[1 for i in range(len(labels_false))]

# precision2,recall2,fbeta_score2,support2=precision_recall_fscore_support(y_t_updated,labels_updated,labels=[1,2,3],average="micro")
# print("precision: ",precision2,"\n","Recall: ",recall2,"\n","Fbeta_score",fbeta_score2,"\n","Support: ",support2)

# precision3=precision_score(y_t_updated,labels_updated,labels=[1,2,3],average="micro")
# recall3=recall_score(y_t_updated,labels_updated,labels=[1,2,3],average="micro")
# print("precision3",precision3)
# print("recall3",recall3)