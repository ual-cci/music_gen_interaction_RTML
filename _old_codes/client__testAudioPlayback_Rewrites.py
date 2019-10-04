# inspiration from https://github.com/spatialaudio/jackclient-python/issues/59

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

DEBUG_simulate_slowdown_pre = True
DEBUG_simulate_slowdown_post = False
VERBOSE_audio_client = True
VERBOSE_messaging_client = False

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


def process(frames):
    previous = np.zeros(blocksize, )

    if qout.empty():
        #print("empty")
        client.outports[0].get_array()[:] = previous
    else:
        #print("we have smth")
        data = qout.get()
        previous = data
        #print("data =", data)
        #print("data", data.shape, data)

        client.outports[0].get_array()[:] = data

# Use queues to pass data to/from the audio backend
queuesize = 4000
blocksize = 1024 # 256, 512 and 1024 are alright
#blocksize = 512

qout = queue.Queue(maxsize=queuesize)
qin = queue.Queue(maxsize=queuesize)
event = Event()

from threading import Thread


class ClientMusic(object):
    PORT = "5000"
    Handshake_REST_API_URL = "http://localhost:" + PORT + "/handshake"
    Audio_REST_API_URL = "http://localhost:" + PORT + "/give_np_audio"


    def requesting_audio(self):
        #audio = np.load("data/saved_audio.npy")
        audio = np.load("data/saved_audio_better.npy")

        k = 0
        while True:
            if DEBUG_simulate_slowdown_pre:
                time.sleep(3.1) # < what if it takes long time??? => Then it's choppy!

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

    samplerate = client.samplerate
    print("samplerate", samplerate)

    client.set_xrun_callback(xrun)
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

    # Pre-fill queues
    data = np.zeros((blocksize,), dtype='float32')

    for k in range(10):
        qout.put_nowait(data) # the output queue needs to be pre-filled

    with client:
        i.connect(capture[0])
        # Connect mono file to stereo output
        o.connect(playback[0])
        o.connect(playback[1])

        TMP = ClientMusic()
        thread = Thread(target=TMP.requesting_audio)
        thread.start()

        while True:
            data = qin.get()
            qout.put(data)

except (queue.Full):
    raise RuntimeError('Queue full')
except KeyboardInterrupt:
    print('\nInterrupted by User')
