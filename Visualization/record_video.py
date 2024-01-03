import cv2
from datetime import datetime

"""
Seems like the cv2 Video Writer is not working on MacOS, 
I spend my afternoon trying different fourcc and changing format with filename, but it just not outputing any video.
Similiar problem link: https://answers.opencv.org/question/236051/videowriter-not-working-on-macos/

Solved:
I added the line of code below, and the problem was gone.
-> cap.set(6, fourcc)

"""

def record_video(video_source=0, duration=2):
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m','j','p','g')

    height = 480
    width = 640
    cap.set(3, height)
    cap.set(4, width)
    cap.set(6, fourcc)

    out = cv2.VideoWriter(f'output{datetime.now()}.mov', fourcc, 20.0, (width, height))

    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

record_video()
