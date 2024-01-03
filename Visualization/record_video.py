import cv2
import time

"""
Seems like the cv2 Video Writer is not working on MacOS, 
I spend my afternoon trying different fourcc and changing format with filename, but it just not outputing any video.
Similiar problem link: https://answers.opencv.org/question/236051/videowriter-not-working-on-macos/

"""

def record_video(duration=2):
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3,480)
    cap.set(4,640)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m','j','p','g')
    cap.set(6, fourcc)

    out = cv2.VideoWriter('output4.mov', fourcc, 20.0, (640, 480))

    # start_time = time.time()
    # while int(time.time() - start_time) < duration:
    while True:
        ret, frame = cap.read()
        # print(frame.shape)
        if ret:
            out.write(frame)  # Write frame to video file
            # Display the frame (optional, can be commented out in production)
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break  # If there is no frame, exit

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

record_video()
