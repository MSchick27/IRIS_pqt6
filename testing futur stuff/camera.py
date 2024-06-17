import time
import cv2

#get available camera indexes

all_camera_idx_available = []

for camera_idx in range(10):
    cap = cv2.VideoCapture(camera_idx)
    if cap.isOpened():
        print(f'Camera index available: {camera_idx}')
        all_camera_idx_available.append(camera_idx)
        cap.release()



#Create an object to read from camera  INDEX 1 should be usb cam
cap=cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
print("Width=", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height=",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#we check if the camera is opened previously or not
if (cap.isOpened()==False):
    print("Error reading video file")

Width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# VideoWriter object will create a frame of the above defined output is stored in 'output.avi' file.
result = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'),30,(640,480))

while(True):
        ret,frame=cap.read()
        result.write(frame)
        cv2.imshow("OpenCVCam", frame)
        
        #Press Q to stop the process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



#When everything is done, release the video capture and videi write objects
cap.release()
result.release()
cv2.destroyAllWindows()
