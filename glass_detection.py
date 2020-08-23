import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from shutil import copyfile
import os, glob
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

framework="tf"  #tf, tflite, trt
model="yolov4"  #yolov3 or yolov4
tiny=False      #yolo or yolo-tiny
iou=0.45        #iou threshold
score=0.25      #score threshold
output='./detections/'  #path to output folder

#def main():
def glass_detector(image_name):
    image_size=416
    imput_image=image_name
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = image_size
    images = [imput_image]

    # load model
    weights_loaded="./checkpoints/glasses-tf-416"
    if framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=weights_loaded)
    else:
            saved_model_loaded = tf.saved_model.load(weights_loaded, tags=[tag_constants.SERVING])

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(images, 1):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        if framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        #image = utils.draw_bbox(original_image, pred_bbox)
        cropped_image = utils.draw_bbox(original_image, pred_bbox)
        # image = utils.draw_bbox(image_data*255, pred_bbox)
        image = Image.fromarray(cropped_image.astype(np.uint8))
        #if not FLAGS.dont_show:
            #image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        cv2.imwrite(output + 'DetectedGlass' + str(count) + '.jpg', image)
        return image

def getUrl(image_path):
    glass_image=glass_detector(image_path)
    #copyfile(image_path, "./detections/DetectedGlass1.jpg")
    print("Glass image detected............")

    test_img_fresh=cv2.imread("./detections/DetectedGlass1.jpg")
    test_img=cv2.resize(test_img_fresh,(200,200))
    test_img = test_img.astype('float32')
    # normalize to the range 0-1
    test_img /= 255.0

    test_image_for_view=np.array(test_img)
    test_image = np.expand_dims(test_image_for_view, axis=0)


    #Load Model
    from tensorflow.keras.models import load_model
    try:
        Parent_model = load_model('trained models\\model_Parent_25_batch.h5')
        print("Parent model loaded successfully.")
    except:
        print("Issue while loading parent model.")

    pred_parent=Parent_model.predict(test_image)
    final_parent_pred=list(np.argmax(pred_parent, axis=1))[0]

    ParentClass=np.where(final_parent_pred==0, "eyeframe",
             (np.where(final_parent_pred==1,"NonPowerReading",
             "sunglasses"))).item()
    try:
        if(ParentClass=="eyeframe"):
            child_model = load_model('trained models\\model_eyeframes_25_batch.h5')
        elif(ParentClass=="NonPowerReading"):
            child_model = load_model('trained models\\model_NonPowerReading_25_batch.h5')
        elif(ParentClass=="sunglasses"):
            child_model = load_model('trained models\\model_Sunglasses_25_batch.h5')
        print("Child model loaded successfully.")
    except:
        print("Issue while loading child model.")


    pred_child=child_model.predict(test_image)
    final_child_pred=list(np.argmax(pred_child, axis=1))[0]

    ChildClass=np.where(final_child_pred==0, "Aviator",
                       (np.where(final_child_pred==1,"Oval",
                       (np.where(final_child_pred==2,"Rectangle",
             "Wayfarer"))))).item()

    output_string=f"A {ParentClass} found with {ChildClass} shape"

    if(ParentClass=="eyeframe"):
        if(ChildClass=="Aviator"):
            url="https://www.lenskart.com/eyeglasses/brands/vincent-chase-eyeglasses/aviator.html"
        elif(ChildClass=="Oval"):
            url="https://www.lenskart.com/eyeglasses/brands/vincent-chase-eyeglasses/vc-round-eyeglasses.html"
        elif(ChildClass=="Rectangle"):
            url="https://www.lenskart.com/eyeglasses/brands/vincent-chase-eyeglasses/rectangle.html"
        elif(ChildClass=="Wayfarer"):
            url="https://www.lenskart.com/eyeglasses/brands/vincent-chase-eyeglasses/wayfarer.html"

    elif(ParentClass=="NonPowerReading"):
        if(ChildClass=="Aviator"):
            url="https://www.lenskart.com/search?q=aviator%20non%20power%20reading"
        elif(ChildClass=="Oval"):
            url="https://www.lenskart.com/search?q=oval%20Non-Power%20Reading"
        elif(ChildClass=="Rectangle"):
            url="https://www.lenskart.com/search?q=Rectangle%20Non-Power%20Reading"
        elif(ChildClass=="Wayfarer"):
            url="https://www.lenskart.com/search?q=Wayfarer%20Non-Power%20Reading"

    elif(ParentClass=="sunglasses"):
        if(ChildClass=="Aviator"):
            url="https://www.lenskart.com/sunglasses/special/vc-aviator.html"
        elif(ChildClass=="Oval"):
            url="https://www.lenskart.com/sunglasses/special/vc-rounders.html"
        elif(ChildClass=="Rectangle"):
            url="https://www.lenskart.com/sunglasses/brands/vincent-chase-sunglasses/rectangle.html"
        elif(ChildClass=="Wayfarer"):
            url="https://www.lenskart.com/sunglasses/brands/vincent-chase-sunglasses/wayfarer.html"
    os.remove("./detections/DetectedGlass1.jpg")
    return output_string,url,ParentClass,ChildClass

