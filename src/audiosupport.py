"""PyAudio Example: Play a WAVE file."""



import pyaudio
import wave

import sys
import os.path


path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../sound/"))


CHUNK = 1024

def play(filename):
    wf = wave.open(filename, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while data != '':
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()

def beep():
    play(os.path.join(path_to_script, "../sound/beep-1.wav"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
        sys.exit(-1)
    filename = sys.argv[1]
    beep()
    #play(filename)

