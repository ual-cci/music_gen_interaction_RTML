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

def get_audio_HANDLERHAXES(lenght = 1600):
    # get audio of this length and return it

    # simulate the way how we will get data
    #buf = np.ones([lenght, blocksize,] )

    buf = []
    for k in range(lenght):
        print(k)
        #data = np.ones(blocksize, )
        #data = np.random.rand(blocksize, )
        #qin.put(data)


        ### I suspect this only gives a handler ...
        datain=client.inports[0].get_array()
        #buf[k,:] = np.asarray(datain)
        buf.append(np.asarray(datain))
    time.sleep(1)

    return buf

tmp_x_frames = np.load("tmp_x_frames.npy")
tmp_y_frames = np.load("tmp_y_frames.npy")
def get_audio_FROMFILE(k=0, lenght = 40):
    # get audio of this length and return it

    # load a sample from file (which we would eventually generate...)
    #print("using:", tmp_x_frames.shape, tmp_y_frames.shape)
    t_k = k % len(tmp_x_frames)
    sample = tmp_x_frames[t_k,:,0:1024]
    print("loaded k=", k, sample.shape)

    #print("sample.shape",sample.shape) # sample.shape (40, 1024)
    return sample

def get_audio(lenght = 40):
    # get audio of this length and return it

    # simulate the way how we will get data
    #buf = np.ones([lenght, blocksize,] )

    buf = []
    for k in range(lenght):
        print(k)
        data = np.ones(blocksize, )
        data = np.random.rand(blocksize, )
        #qin.put(data)


        ### I suspect this only gives a handler ...
        datain=client.inports[0].get_array()
        qin.put(datain)

        #buf[k,:] = np.asarray(datain)
        #buf.append(np.asarray(datain))
    #time.sleep(1)

    return buf

    #for sample in buf:
    #    qin.put(sample)



def process(frames):
    previous = np.zeros(blocksize, )

    if qout.empty():
        print("empty")
        client.outports[0].get_array()[:] = previous
    else:
        print("we have smth")
        data = qout.get()
        previous = data
        print("data", data.shape, data)

        client.outports[0].get_array()[:] = data

# Use queues to pass data to/from the audio backend
queuesize = 4000
blocksize = 1024

qout = queue.Queue(maxsize=queuesize)
qin = queue.Queue(maxsize=queuesize)
event = Event()

from threading import Thread


class ClientMusic(object):
    PORT = "5000"
    Handshake_REST_API_URL = "http://localhost:" + PORT + "/handshake"
    Audio_REST_API_URL = "http://localhost:" + PORT + "/give_np_audio"


    def requesting_audio(self):

        #buf = self.client_audio_query()

        k = 0
        while True:
            k += 1
            buf = get_audio()
            #buf = get_audio_FROMFILE(k)
            if VERBOSE_audio_client:
                print("got batch of ", len(buf))
            for sample in buf:
                qin.put(sample)

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
            # on thread:
            #get_audio()

            #if qin.empty():
            #    time.sleep(1)

            data = qin.get()
            qout.put(data)

except (queue.Full):
    raise RuntimeError('Queue full')
except KeyboardInterrupt:
    print('\nInterrupted by User')
