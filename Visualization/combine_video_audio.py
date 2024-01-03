from moviepy.editor import VideoFileClip, AudioFileClip

def combine_audio_video(audio_filename, video_filename, output_filename="output_with_audio.mp4"):
    video_clip = VideoFileClip(video_filename)
    audio_clip = AudioFileClip(audio_filename)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    video_filename = "output_video3.mov"
    audio_filename = "output_audio3.wav"

    combine_audio_video(audio_filename, video_filename, "final_output.mov")

    print("Recording complete. File saved as 'final_output.mp4'")
