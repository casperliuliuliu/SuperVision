from combine_video_audio import combine_audio_video
from record_video import record_video
from record_audio import record_audio
import threading

"""
[ Unfix ]
Threading might cause a error:
"terminating due to uncaught exception of type NSException".

It seems like a problem when multithreading on MacOS. (https://stackoverflow.com/questions/63202212/why-python-multithreading-runs-like-a-single-thread-on-macos)


"""

if __name__ == "__main__":

    video_filename = "output_video3.mov"
    audio_filename = "output_audio3.wav"

    duration = 5  # seconds
    video_thread = threading.Thread(target=record_video, args=(duration, video_filename))
    # audio_thread = threading.Thread(target=record_audio, args=(duration, audio_filename))

    video_thread.start()
    # audio_thread.start()
    video_thread.join()

    # combine_audio_video(audio_filename, video_filename, "final_output.mov")

    print("Recording complete. File saved as 'final_output.mp4'")