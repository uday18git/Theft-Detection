# import threading 
# import cv2
# from deepface import DeepFace

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
# counter=0
# face_match = False
# reference_img=cv2.imread("reference.jpg")
# reference_img2=cv2.imread("reference2.jpg")
# val=-1
# def check_face(frame):
#     global face_match
#     try:
#         if DeepFace.verify(frame,reference_img.copy())['verified']:
#             face_match=True
#             val=1
#         elif DeepFace.verify(frame,reference_img2.copy())['verified']:
#             face_match=True
#             val=2
#         # elif DeepFace.verify(frame,reference_img3.copy())['verified']:
#         #     face_match=True
#         else:
#             face_match=False
#     except ValueError:
#         face_match=False
# while True:
#     ret,frame = cap.read()

#     if ret:
#         if counter % 30 == 0:
#             try:
#                 threading.Thread(target=check_face, args=(frame.copy(),)).start()
#             except ValueError:
#                 pass
#         counter+=1
#         if face_match:                                        #font size,green,thickness
#             if val==1:
#                 cv2.putText(frame,"elon",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
#             elif val==2:
#                 cv2.putText(frame,"bill gates",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                
#         else:
#             cv2.putText(frame,"NO MATCH!",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)

#         cv2.imshow("video",frame)

#     key = cv2.waitKey(1)
#     if key==ord("q"):
#         break

# cv2.destroyAllWindows()
import threading 
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
counter = 0
face_match = False
reference_img = cv2.imread("reference.jpg")
reference_img2 = cv2.imread("reference2.jpg")
val = -1

def check_face(frame):
    global face_match, val
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
            val = 1
        elif DeepFace.verify(frame, reference_img2.copy())['verified']:
            face_match = True
            val = 2
        else:
            face_match = False
            val = -1
    except ValueError:
        face_match = False
        val = -1

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1
        if face_match:
            if val == 1:
                cv2.putText(frame, "elon", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif val == 2:
                cv2.putText(frame, "bill gates", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()


