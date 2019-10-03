# inspiration from https://github.com/spatialaudio/jackclient-python/issues/59
# 1.) play a sound into output

import numpy as np
try:
    import queue  # Python 3.x
except ImportError:
    import Queue as queue  # Python 2.x

queuesize = 4000
qout = queue.Queue(maxsize=queuesize)
qin = queue.Queue(maxsize=queuesize)

# Process which gets data from qout to the waiting sound client (jackd)
def process(frames):
    nothing = np.zeros(blocksize, )

    if qout.empty():
        print("empty")
        client.outports[0].get_array()[:] = nothing
    else:
        data = qout.get()

        if len(data) == 0:
            print("empty")
            client.outports[0].get_array()[:] = nothing
            return 0

        #print("data", data.shape, data)
        #print("stats", np.min(data), np.mean(data), np.max(data))

        client.outports[0].get_array()[:] = data

# Use queues to pass data to/from the audio backend
queuesize = 4000
blocksize = 1024


# Basic setup for the audio player in Python
import jack
client = jack.Client("thru_client")
client.blocksize = 1024
samplerate = client.samplerate

client.set_process_callback(process)

client.inports.register('in_{0}'.format(1))
client.outports.register('out_{0}'.format(1))
i = client.inports[0]
capture = client.get_ports(is_physical=True, is_output=True)
playback = client.get_ports(is_physical=True, is_input=True, is_audio=True)
o = client.outports[0]

timeout = blocksize / samplerate
print("Processing input in %d ms frames" % (int(round(1000 * timeout))))

# Pre-fill queues
data = np.zeros((blocksize,), dtype='float32')

for k in range(1):
    qout.put_nowait(data)  # the output queue needs to be pre-filled


# 2.) load processed sound (in the same way as we will generate sounds)
# we started with spectrograms
DEMO_LOAD_FROM_SPECTROGRAMS = True
DEMO_MAKE_FROM_WAV_FILE = False


if DEMO_LOAD_FROM_SPECTROGRAMS:
    sample_rate = 44100
    #fft_settings = [2048, 1024, 512]
    fft_settings = [2048, 2024, 512] # test with longer window - should have better resolution in frq. but worse in time
    # sounds better obv...

    tmpname = "test1_withWindowSize2048"

    fft_size = fft_settings[0]
    window_size = fft_settings[1]
    hop_size = fft_settings[2]
    sequence_length = 40
    sample_rate = 44100

    if DEMO_MAKE_FROM_WAV_FILE:
        from utils.audio_dataset_generator import AudioDatasetGenerator
        dataset = AudioDatasetGenerator(fft_size, window_size, hop_size, sequence_length, sample_rate)
        print("loading dataset from a wav file")
        audio_data_path = "/media/vitek/Data/Vitek/Projects/2019_LONDON/music generation/small_file/"
        dataset.load(audio_data_path, force=True, prevent_shuffling=True)

        tmp_y_frames = dataset.y_frames[0:6000]
        #import numpy as np
        #np.save("data/tmp_y_frames_6000NoShuffle.npy", dataset.y_frames[0:6000])
    else:
        import numpy as np
        tmp_y_frames = np.load("data/tmp_y_frames_6000NoShuffle.npy")

    print("using converted spectrograms")
    print("tmp_y_frames hard loaded:", tmp_y_frames.shape)

    def griffin_lim(stftm_matrix, max_iter=100):
        """"Iterative method to 'build' phases for magnitudes."""

        fft_size = fft_settings[0]
        window_size = fft_settings[1]
        hop_size = fft_settings[2]
        stft_matrix = np.random.random(stftm_matrix.shape)
        y = librosa.core.istft(stft_matrix, hop_size, window_size)
        for i in range(max_iter):
            stft_matrix = librosa.core.stft(y, fft_size, hop_size, window_size)
            stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
            y = librosa.core.istft(stft_matrix, hop_size, window_size)
        return y

    window_size = 1024
    griffin_iterations = 60

    predicted_magnitudes = np.asarray(tmp_y_frames[500])
    #predicted_magnitudes = np.zeros(1025, )
    for prediction in tmp_y_frames[500+1:2500+1]:
        predicted_magnitudes = np.vstack((predicted_magnitudes, prediction))

    predicted_magnitudes = np.array(predicted_magnitudes).reshape(-1, window_size + 1)
    print("predicted_magnitudes", predicted_magnitudes.shape)
    # predicted_magnitudes (2001, 1025)

    import librosa
    # and convert the spectrograms to audio signal
    audio = [griffin_lim(predicted_magnitudes.T, griffin_iterations)]
    audio = np.asarray(audio[0])
    print("audio.shape", audio.shape)
    # window size 2048 => audio.shape (1024000,)

    ###### or directly reload audio signal:
    """
    audio = np.load("data/tmp_audio_reconstructed.npy")
    audio = np.asarray(audio[0])
    print("audio hard loaded:", audio.shape)
    """

    sample_rate = 44100
    print("saving the audio file for inspection into data/testconcept2_testing"+tmpname+".wav")
    librosa.output.write_wav('data/testconcept2_testing_'+tmpname+'.wav', audio, sample_rate)

def get_audio_random():
    data = np.random.rand(blocksize, )
    qin.put(data)

def get_audio_capture():
    datain=client.inports[0].get_array()
    qin.put(datain)

def get_audio_part_from_reconstructed_file(lenght = 1024, k=0):
    data = audio[lenght*k:lenght*(k+1)]
    qin.put(data)


# Simple example, the audio client finally starts and is fed one by one what to play
#  - we get audio (generate/random/load) to qin
#  - this one sample of 1024 is taken from qin and put into qout
#  - client calls a process function of it (taking it from qout into the audio buffer)

k = 0
with client:
    i.connect(capture[0])
    # Connect mono file to stereo output
    o.connect(playback[0])
    o.connect(playback[1])

    while True:
        # Each one of these saves only one batch of audio of the size of 1024 samples
        if DEMO_LOAD_FROM_SPECTROGRAMS:
            get_audio_part_from_reconstructed_file(lenght=1024, k=k)
        else:
            get_audio_random()
            #get_audio_capture()

        data = qin.get()
        qout.put(data)

        k += 1
