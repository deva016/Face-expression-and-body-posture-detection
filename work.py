from flask import Flask,request, render_template
from keras.models import load_model
from time import sleep
from keras.preprocessing import image
from keras_preprocessing.image import img_to_array
import cv2
import datetime
import mediapipe as mp
import numpy as np
# app=Flask(__name__)

# @app.route('/')
def login():
    face_classifier=cv2.CascadeClassifier(r'D:\project\pyhon\MCP Project\Face expression detection\haarcascade_frontalface_default.xml')
    classifier=load_model('expressiondata.h5')
    #assign
    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose
    count=0
    total=0
    class_labels=['20','80','70','0','20']#angry,happy,sad,neutral,suprise
    def calculateangle(a,b,c):
        a=np.array(a)
        b=np.array(b)
        c=np.array(c)

        radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
        angle=np.abs((radians*180.0/np.pi))

        if angle>180.0:
            angle=360-angle
        return angle

    def rescale_frame(frame, percent=75):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

    if count==0:
        cap=cv2.VideoCapture(0)
        with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret,frame=cap.read()
                frame = rescale_frame(frame, percent=160)
                image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                image.flags.writeable=False
                original_frame = frame.copy()
                result=pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                landmarks=result.pose_landmarks.landmark

                sholder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle=calculateangle(sholder,elbow,wrist)

                s = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                e = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                w = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                a = calculateangle(s, e, w)

                left_elob = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_sholder= [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip= [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ang= calculateangle(left_elob,left_sholder,left_hip)

                cv2.putText(image,str(angle),tuple(np.multiply(elbow,[640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(image,str(a),tuple(np.multiply(e, [640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA)
                cv2.putText(image, str(ang),tuple(np.multiply(left_sholder, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2, cv2.LINE_AA)
                # print(mp_drawing)

                if angle>60.0 and angle<80.0 and a>60.0 and a<80.0:
                    count=10
                    cv2.imwrite("selfi.png", original_frame)
                    break;
                elif ang>150.0 and ang<170 and angle>=30 and angle<=40:
                    count=20
                    cv2.imwrite("selfi.png", original_frame)
                    break;
                #     cv2.imwrite('selfi.png',frame)

                #rendering
                mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66),thickness=2 , circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))

                cv2.imshow('Mediapipe Feed',image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break;
            cap.release()

    c=count
    if c==10 or count==20:
        cap=cv2.VideoCapture(0)
        while True:
            ret,frame=cap.read()
            frame = rescale_frame(frame, percent=160)
            original_frame=frame.copy()
            labels=[]
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_classifier.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                time_slot = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                file = f'selfie-{time_slot}.png'
                roi_gray=gray[y:y+h,x:x+w]
                roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray])!=0:
                   roi=roi_gray.astype('float')/255.0
                   roi=img_to_array(roi)
                   roi=np.expand_dims(roi,axis=0)

                   preds=classifier.predict(roi)[0]
                   label=class_labels[preds.argmax()]
                   label_position=(x,y)

                   cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

                   if int(label)>50:
                       cv2.imwrite(file,original_frame)
                       c=c+int(label)
                       print(c)
                       break;

                else:
                    cv2.putText(frame,'No Face Found',label_position,(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                    # if c>0:
                    #     cv2.imwrite("selfi.png", original_frame)
                    #     total=c+80


            cv2.imshow('Emotion Detection' ,frame)
            if cv2.waitKey(1) & 0xFF==ord('q') or c>=70:
                break;
        cap.release()
    print(c)
    return 'done'+str(c)

if __name__=="__main__":
    app.run(debug=True)