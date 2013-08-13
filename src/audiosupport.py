"""PyAudio Example: Play a WAVE file."""



import pyaudio
import wave
import math
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

def play2():
    PyAudio = pyaudio.PyAudio
    RATE = 16000
    WAVE = 1000
    data = ''.join([chr(int(math.sin(x/((RATE/WAVE)/math.pi))*127+128)) for x in xrange(RATE)])
    p = PyAudio()

    stream = p.open(
            format=p.get_format_from_width(1),
            channels=1,
            rate=RATE,
            output=True
            )
    for DISCARD in xrange(5):
            stream.write(data)
            stream.stop_stream()
            stream.close()
            p.terminate()

def beep():
    #play2()
    play(os.path.join(path_to_script, "../sound/beep-1.wav"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
        sys.exit(-1)
    filename = sys.argv[1]
    beep()
    #play(filename)

