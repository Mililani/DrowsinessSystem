#mar (mouth.aspect.ratio) ear(eye.aspect.ratio)
import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import winsound


EYE_AR_THRESH = 0.3#if the aspect ration falls bellow this it will be seen as a blink
EYE_AR_CONSEC_FRAMES = 7#if nine successive frames either ear less than or lar greater then it will be recorded as a sign of drowsiness
MOUTH_AR_THRESH = 0.5

SHOW_POINTS_FACE = False#landmark point will be set to false so you have screen with user only
SHOW_CONVEX_HULL_FACE = False#convex hull will be set to zero to see user eyes 
SHOW_INFO = False#information regarding counters will be set to zero

ear = 0
mar = 0

CounterFRAMES_EYE = 0
CounterFRAMES_Lips = 0
CounterBLINK = 0
CounterLips = 0
#the above counter will help with the total number successive frames that will have an E.A.R less than EYE_EAR_Threshold and L.A.R greater then Lips_lar_threshold
def eye_aspect_ratio(eye):#this code represents the vertical distance of the eye
    d1 = dist.euclidean(eye[1], eye[5])
    d2 = dist.euclidean(eye[2], eye[4])
    d3 = dist.euclidean(eye[0], eye[3])
    return (d1 + d2) / (2.0 * d3)#E.A.R calculation for one side

def mouth_aspect_ratio(mouth):
    d1 = dist.euclidean(mouth[5], mouth[8])
    d2 = dist.euclidean(mouth[1], mouth[11])	
    d3 = dist.euclidean(mouth[0], mouth[6])
    return (d1 + d2) / (2.0 * d3) 




videoSteam = cv2.VideoCapture(0)#capture vebcame
ret, frame = videoSteam.read()
size = frame.shape#image size 

detector = dlib.get_frontal_face_detector()#this will help get the facial land markings needed
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")#code will help predict lanmarks on face
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]#landmarks index point from left to right 
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

model_points = np.array([(0.0, 0.0, 0.0),
                         (0.0, -330.0, -65.0),        
                         (-225.0, 170.0, -135.0),     
                         (225.0, 170.0, -135.0),      
                         (-150.0, -150.0, -125.0),    
                         (150.0, -150.0, -125.0)])

focal_length = size[1]#this will help us when it come to the angel which will be caputerd from our webcam and the manification of how large the person  will be
center = (size[1]/2, size[0]/2)

cam_matrix = np.array([[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double")#this code part will be used to help describe the mapping of our webcam from 3d to 24 point image

distance_coeffs = np.zeros((4,1))#represents the distance coefficient which will later be projected

t_end = time.time()#help use track the number of seconds which has passed since epoch (time start)
while(True):
    ret, frame = videoSteam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#this part represents the turn of the image to gray scale
    rects = detector(gray, 0)#this will be the detection of the frame in grayscale formate

    for rect in rects:#this part of the code will over over each of the frames and then apply facial landmark detection for each of them ,as well as determine facial landmarks
        shape = predictor(gray, rect)#retctengle shapes in the frame 
        shape = face_utils.shape_to_np(shape)
        left = shape[lStart:lEnd]#leftt eye frame start position and and position
        right = shape[rStart:rEnd]#right eye frame start position and and position
        jaw = shape[48:61]

        left_EAR = eye_aspect_ratio(left)#leftEar is give now a value/calclation
        right_EAR = eye_aspect_ratio(right) #leftEar is give now a value/calclation
        ear = (left_EAR + right_EAR) / 2.0#avarage calculation of the E.A.R for both sides
        mar = mouth_aspect_ratio(jaw)

        image_points = np.array([
                                (shape[30][0], shape[30][1]),
                                (shape[8][0], shape[8][1]),
                                (shape[36][0], shape[36][1]),
                                (shape[45][0], shape[45][1]),
                                (shape[48][0], shape[48][1]),
                                (shape[54][0], shape[54][1])
                                ], dtype="double")#this will be the images points we find on our webcam
                                                    #Which will measure from the center of one hairline to the chin tip /left right of the face/right to left


        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, cam_matrix, distance_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, cam_matrix, distance_coeffs)#postion the image

        if SHOW_POINTS_FACE:
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        point1 = (int(image_points[0][0]), int(image_points[0][1]))
        point2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        if SHOW_CONVEX_HULL_FACE: #this part of the code helps visualizing the facil landmarks for eye regions  
            left_Eye_Hull = cv2.convexHull(left)
            right_Eye_Hull = cv2.convexHull(right)
            jaw_Hull = cv2.convexHull(jaw)

            

            cv2.drawContours(frame, [left_Eye_Hull], 0, (255, 255, 255), 1)# code will draw  the boundary points of the eyes convex hull 
            cv2.drawContours(frame, [right_Eye_Hull], 0, (255, 255, 255), 1)
            cv2.drawContours(frame, [jaw_Hull], 0, (255, 255, 255), 1)
            cv2.line(frame, point1, point2, (255,255,255), 2)#this will help us draw lines image coordenates 


        if point2[1] > point1[1]*1.5 or CounterBLINK > 5 or CounterLips > 3:# this part of the code will determine if a user is sleepy if any of the conditions are true arlet will be sent
            cv2.putText(frame, "Alert Driver Drowsy!", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        if ear < EYE_AR_THRESH:
            CounterFRAMES_EYE += 1

            if CounterFRAMES_EYE >= EYE_AR_CONSEC_FRAMES:#this part of the code will  tell you that driver is tired as the COUNTER_FRAMES_EYE is less than equal to EYE_EAR_CONSEC_FRAMES which means drive frames are below wat the must be 
                cv2.putText(frame, " Driver Asleep!", (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(449, 500)#beep to alret driver that he she is sleeping
        else:
            if CounterFRAMES_EYE > 2:
                CounterBLINK += 1
            CounterFRAMES_EYE = 0#normal blink
        
        if mar >= MOUTH_AR_THRESH:
            CounterFRAMES_Lips += 1
        else:
            if CounterFRAMES_Lips > 5:
                CounterLips += 1
      
            CounterFRAMES_lips = 0#normal yawn 
        
        if (time.time() - t_end) > 60:#rest the blink number and count number back to 0
            t_end = time.time()
            CounterBLINK = 0
            CounterLips = 0
        
    if SHOW_INFO:#this part of the code  will help draw the yawns and blink (EAR and LAR)
        cv2.putText(frame, "E.A.R: {:.2f}".format(ear), (30, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "M.A.R: {:.2f}".format(mar), (200, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Blinks: {}".format(CounterBLINK), (10, 30),#display blink counter/number of blinks 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Lips: {}".format(CounterLips), (10, 60),#lip counter number of times yawned 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Output", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('p'):
        SHOW_POINTS_FACE = not SHOW_POINTS_FACE#display land markpoint of face 
    if key == ord('c'):
        SHOW_CONVEX_HULL_FACE = not SHOW_CONVEX_HULL_FACE#display eye covex hull 
    if key == ord('i'):
        SHOW_INFO = not SHOW_INFO#display counter information of ear,lips ,mar
    time.sleep(0.02)
    
videoSteam.release()  
cv2.destroyAllWindows()
