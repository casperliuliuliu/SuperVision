import cv2
from datetime import datetime
import time

"""
[ Fixed ]
Seems like the cv2 Video Writer is not working on MacOS, 
I spend my afternoon trying different fourcc and changing format with filename, but it just not outputing any video.
Similiar problem link: https://answers.opencv.org/question/236051/videowriter-not-working-on-macos/

Solved:
I added the line of code below, and the problem was gone.
-> cap.set(6, fourcc)

"""

def record_video(duration=-1, filename=None, video_source=0):
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

    if filename == None:
        filename = f"output{datetime.now()}.mov"

    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

    start_time = time.time()
    while int(time.time() - start_time) < duration:
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
if __name__ == "__main__":
    record_video(5, "output2.mov")
