import pyaudio
import wave

def record_audio(duration, filename="output_audio.wav"):
    chunk = 1024  # Record in chunks
    format = pyaudio.paInt16  # 16-bit format
    channels = 1  # mono
    sample_rate = 44100  # Sample rate
    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []

    for _ in range(int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    record_audio(2)