# inspiration from https://github.com/spatialaudio/jackclient-python/issues/59
# start jackd with ./run_jackd.sh
# restart with: killall jackd

from __future__ import division, print_function
import time
import jack
import sys
import numpy as np
from threading import Event
try:
    import queue  # Python 3.x
except ImportError:
    import Queue as queue  # Python 2.x
from threading import Thread
from timeit import default_timer as timer
import requests
from oscpy.server import OSCThreadServer
from time import sleep
import librosa
from scipy import signal

DEBUG_simulate_slowdown_pre = False
DEBUG_simulate_slowdown_post = False
VERBOSE_audio_client = True
VERBOSE_queues_status = False

def print_error(*args):
    print(*args, file=sys.stderr)

def xrun(delay):
    print_error("An xrun occured, increase JACK's period size?")

def shutdown(status, reason):
    print_error('JACK shutdown!')
    print_error('status:', status)
    print_error('reason:', reason)
    event.set()

def stop_callback(msg=''):
    if msg:
        print_error(msg)
    for port in client.outports:
        port.get_array().fill(0)
    event.set()

def save_audio_debug(name,audio,sample_rate):
    import scipy.io.wavfile
    scipy.io.wavfile.write("data/"+name+".wav", sample_rate, audio)


"""
def get_audio_FROMFILE(audio, k=0, lenght = 1024):
    t_k = k # % len(audio)
    sample = np.asarray(audio[t_k * lenght:(t_k + 1) * lenght])
    #print(sample.shape)
    return sample # (1024,)

def get_audio(blocksize = 1024):
    data = np.random.rand(blocksize, )
    return data # (1024,)

def get_audio_HandlerOnly():
    datain = client.inports[0].get_array() # automatically knows the lenghts I guess
    return datain

def get_audio_chunk_from_server_HAX():
    # hax
    audio_chunk = np.random.rand(36352, ) #(36352,)
    return audio_chunk
"""

def process(frames):
    previous = np.zeros(blocksize, )
    if VERBOSE_queues_status:
        print("qout", qout.qsize(), "/", queuesize)

    if qout.empty():
        print("empty, waiting with nothing!")
        client.outports[0].get_array()[:] = previous
    else:
        #print("we have smth")
        data = qout.get()
        previous = data
        #print("data =", data)
        #print("data", data.shape, data)

        client.outports[0].get_array()[:] = data

# Use queues to pass data to/from the audio backend
queuesize = 1000
blocksize = 1024 # 256, 512 and 1024 are alright
#blocksize = 512

qout = queue.Queue(maxsize=queuesize)
qin = queue.Queue(maxsize=queuesize)
event = Event()

CLIENT_sample_rate = 22050

OSC_address = '0.0.0.0'
OSC_port = 8000
OSC_bind = b'/send_i'

# global SIGNAL_interactive_i
SIGNAL_interactive_i = 0.0
SIGNAL_model_i = 0 #< if i want it to start with it, hardcode it here for now
SIGNAL_song_i = 0

SIGNAL_requested_lenght = 128

class ClientMusic(object):
    PORT = "5000"
    Handshake_REST_API_URL = "http://localhost:" + PORT + "/handshake"
    Handshake_GETAUDIO_API_URL = "http://localhost:" + PORT + "/get_audio"
    #sample_rate = 44100
    sample_rate = 22050

    def setup_server_connection(self):

        payload = {"client": "client", "backup_name": "Bob"}
        print("trying at ", self.Handshake_REST_API_URL)
        r = requests.post(self.Handshake_REST_API_URL, files=payload).json()
        print("Handshake request data", r)

        # GET SERVER SETTINGS PROBABLY!
        # OR SETUP SERVER ALSO HERE
        # sample_rate = from response ...

    def audio_from_server(self, requested_lenght):
        t_start_request = timer()
        payload = {"requested_length": str(requested_lenght),
                   "interactive_i": str(SIGNAL_interactive_i),
                   "model_i": str(SIGNAL_model_i),
                   "song_i": str(SIGNAL_song_i),
                   }

        r = requests.post(self.Handshake_GETAUDIO_API_URL, files=payload).json()
        #print("Get audio request data", r)

        audio_response = r["audio_response"]
        t_predict = r["time_predict"]
        t_reconstruct = r["time_reconstruct"]
        t_decode = r["time_decode"]

        audio_response = np.asarray(audio_response)

        t_end_request = timer()

        t_server_total = r["time_server_total"]
        t_client_total = t_end_request-t_start_request

        t_communication = t_client_total - t_server_total # both ways communication included
        seconds_playback = np.max([len(audio_response) / self.sample_rate])
        print("     [L=", str(requested_lenght).center(5), "] audio_response:", audio_response.shape, "time: ", t_client_total, "<?> play", seconds_playback, "sec ..", end='', flush=True)

        return audio_response

    def requesting_audio(self):
        #audio = np.load("data/saved_audio.npy")
        #audio = np.load("data/saved_audio_better.npy")

        k = 0
        t = 0
        last_bit = []
        while True:
            t += 1

            # qin is being cleaned all the time...
            print("qin", qin.qsize(), "/", queuesize)
            # if qout is almost full, just wait ... !
            # or just wait when you have something to play
            global SIGNAL_requested_lenght

            while qout.qsize() > SIGNAL_requested_lenght: # maybe use that?
                time.sleep(0.05)

            if DEBUG_simulate_slowdown_pre:
                time.sleep(3.1) # < what if it takes long time??? => Then it's choppy!

            audio_response = self.audio_from_server(SIGNAL_requested_lenght)

            # Resample:
            #audio_response = librosa.resample(audio_response, 22050, 48000)
            #print(audio_response.shape)
            audio_response = signal.resample(audio_response, len(audio_response)*2) # 22050 of model * 2
            #print(audio_response.shape)

            name = "_"+str(t).zfill(3)
            #save_audio_debug(name, audio_response, self.sample_rate)

            # now we have to split this long signal into batches
            # (bonus) and also keep the last one for the next loop
            l = 0
            batch_arr = []

            if len(last_bit) > 0:
                # concat previous last_bit to audio_response:
                #print("concat", np.asarray(last_bit).shape, "to", audio_response.shape)
                audio_response = np.concatenate((last_bit, audio_response))
                #print("into", np.asarray(audio_response).shape)

            len_of_audio_response = len(audio_response)

            while (l+1)*blocksize <= len_of_audio_response:
                sample = np.asarray(audio_response[l * blocksize:(l + 1) * blocksize])
                #print("l=",l,"sample=",sample.shape, "[",l * blocksize, "-", (l + 1) * blocksize ,"]")
                l += 1

                batch_arr.append(sample)

            # last bit is in the rest
            last_bit = np.asarray(audio_response[l * blocksize:])
            #print("last bit with l=", l, "sample=", last_bit.shape, "[", l * blocksize, "- end (36350)", "]")

            """
            batch_size = 100
            batch_arr = []
            for batch_i in range(batch_size):

                k += 1
                if k > 215:  # with k == 219 this was the end of the sample
                    k = 0  # hax
                buf = get_audio_FROMFILE(audio, k)

                #buf = get_audio(blocksize)
                #buf = get_audio_HandlerOnly()

                batch_arr.append(buf)
            """

            if DEBUG_simulate_slowdown_post:
                time.sleep(0.1) # < what if it takes long time??? => Then it's choppy!

            if VERBOSE_audio_client:
                print("got batch of ", len(batch_arr), "times", len(batch_arr[0]))

            for batch_buf in batch_arr:
                qin.put(batch_buf)

try:
    # Initialise jackd client
    client = jack.Client("thru_client")
    client.blocksize = blocksize
    print("blocksize", blocksize)

    # I can start it with 44100 or 48000 but not 22050 (!) ... make it resample (aka samling rate conversion)?
    samplerate = client.samplerate
    print("samplerate", samplerate, "we want", CLIENT_sample_rate)
    #assert (int(samplerate)) == CLIENT_sample_rate

    # JACKD
    #client.set_xrun_callback(xrun)
    client.set_shutdown_callback(shutdown)
    client.set_process_callback(process)

    client.inports.register('in_{0}'.format(1))
    client.outports.register('out_{0}'.format(1))
    i=client.inports[0]
    capture = client.get_ports(is_physical=True, is_output=True)
    playback = client.get_ports(is_physical=True, is_input=True, is_audio=True)
    o=client.outports[0]

    timeout = blocksize / samplerate
    print("Processing input in %d ms frames" % (int(round(1000 * timeout))))

    # OSC - Interactive listener

    def callback(*values):
        global SIGNAL_interactive_i
        global SIGNAL_model_i
        global SIGNAL_song_i
        global SIGNAL_requested_lenght
        print("OSC got values: {}".format(values))
        # [percentage, model_i, song_i]
        percentage, model_i, song_i, requested_lenght = values

        SIGNAL_interactive_i = float(percentage)/1000.0 # 1000 = 100% = 1.0
        SIGNAL_model_i = int(model_i)
        SIGNAL_song_i = int(song_i)
        SIGNAL_requested_lenght = int(requested_lenght)

    print("Also starting a OSC listener at ",OSC_address,OSC_port,OSC_bind, "to listen for interactive signal (0-1000).")
    osc = OSCThreadServer()
    sock = osc.listen(address=OSC_address, port=OSC_port, default=True)
    osc.bind(OSC_bind, callback)
    #sleep(1000)
    # at the end?

    # Pre-fill queues
    data = np.zeros((blocksize,), dtype='float32')

    for k in range(100):
        qout.put_nowait(data) # the output queue needs to be pre-filled

    with client:
        i.connect(capture[0])
        # Connect mono file to stereo output
        o.connect(playback[0])
        o.connect(playback[1])

        music = ClientMusic()
        music.setup_server_connection()

        thread = Thread(target=music.requesting_audio)
        thread.start()

        while True:
            data = qin.get()
            qout.put(data)

    osc.stop()

except (queue.Full):
    raise RuntimeError('Queue full')
except KeyboardInterrupt:
    print('\nInterrupted by User')
