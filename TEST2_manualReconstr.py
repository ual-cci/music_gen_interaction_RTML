audio_file = "new_audio_samples/dnb/_dnb.wav"

from utils.audio_dataset_generator import AudioDatasetGenerator
import scipy.io.wavfile
import numpy as np

sample_rate          = 44100
fft_settings         = [2048, 1024, 512]
fft_size             = fft_settings[0]
window_size          = fft_settings[1]
hop_size             = fft_settings[2]
sequence_length = 40
sample_rate = 44100
griffin_iterations = 60
import lws
lws_processor = lws.lws(window_size, hop_size, L=5, fftsize=fft_size, mode="music")

method="Griff"
#method="LWS"

from audio_handler import AudioHandler
import librosa

file = audio_file
fft_frames = []
x_frames = []
y_frames = []

if True:
    data, sample_rate = librosa.load(file, sr=sample_rate,
                                     mono=True)
    data = np.append(np.zeros(window_size * sequence_length), data)
    mags_phases = librosa.stft(data, n_fft=fft_size, win_length=window_size, hop_length=hop_size,

                               center=False

                               )
    magnitudes, phases = librosa.magphase(mags_phases)
    for magnitude_bins in magnitudes.T:
        fft_frames += [magnitude_bins]

start = 0
end = len(fft_frames) - sequence_length - 1
step = 1
for i in range(start, end, step):
    done = int(float(i) / float(end) * 100.0)

    x = fft_frames[i:i + sequence_length]
    y = fft_frames[i + sequence_length]
    x_frames.append(x)
    y_frames.append(y)

x_frames = np.asarray(x_frames, dtype=np.float32)
y_frames = np.asarray(y_frames, dtype=np.float32)
print("Loaded: x_frames", x_frames.shape, "y_frames", y_frames.shape)


print("recreating with method", method)


def griffin_lim(stftm_matrix, max_iter=100):
    global hop_size, window_size, fft_size
    """"Iterative method to 'build' phases for magnitudes."""
    stft_matrix = np.random.random(stftm_matrix.shape)
    y = librosa.core.istft(stft_matrix, hop_size, window_size,

                           center=False

                           )

    if not np.isfinite(y).all():
        print("Problem with the signal - it's not finite (contains inf or NaN)")
        print("Signal = ", y)
        y = np.nan_to_num(y)
        print("Attempted hacky fix")

    for i in range(max_iter):
        if not np.isfinite(y).all():
            print("Problem with the signal - it's not finite (contains inf or NaN), in iteration", i)
            print("Signal = ", y)
            y = np.nan_to_num(y)
            print("Attempted hacky fix inside the iterative method")

        stft_matrix = librosa.core.stft(y, fft_size, hop_size, window_size,

                                        center=False

                                        )
        stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
        y = librosa.core.istft(stft_matrix, hop_size, window_size,

                               center=False

                               )
    return y


def lws_reconstruct(data):
    global lws_processor
    X0 = data
    ## HAX, doesnt influence it too much tho
    if X0.dtype != "float64":
        X0 = np.asarray(X0, dtype=np.float64)

    X1 = lws_processor.run_lws(
        X0)  # reconstruction from magnitude (in general, one can reconstruct from an initial complex spectrogram)
    print('{:6}: {:5.2f} dB'.format('LWS', lws_processor.get_consistency(X1)))

    represented = X1
    # now reconstruct from:
    print("\trepresented shape", represented.shape)

    reconstruction = lws_processor.istft(represented)  # where x is a single-channel waveform
    reconstruction = np.asarray(reconstruction)
    print("\toutput reconstruction:", reconstruction.shape)  # (531968,)
    print("\treconstruction data type:", reconstruction.dtype)

    return reconstruction


def spectrogram2audio(spectrogram, method="Griff"):
    global griffin_iterations
    if method == "Griff":
        audio = griffin_lim(spectrogram.T, griffin_iterations)
        audio = np.array(audio)

    elif method == "LWS":
        audio = lws_reconstruct(spectrogram)
        audio = np.array(audio)

    if not np.isfinite(audio).all():
        print("Problem with the signal - it's not finite (contains inf or NaN)")
        print("Signal = ", audio)
        audio = np.nan_to_num(audio)
        print("Attempted hacky fix")

    return audio


def recreate_samples(x_frames, y_frames, method, idx_start=0, amount_samples=1, sequence_max_length=2000,
                     griffin_iterations=100):
    """Generates samples in the supplied folder path."""
    all_audio = []
    if True:
        for i in range(amount_samples):
            random_index = 0
            impulse = x_frames[idx_start + random_index]
            predicted_magnitudes = impulse
            for j in range(sequence_max_length):
                # prediction.shape (1, 1025)
                prediction = y_frames[idx_start + j + 1]
                prediction = prediction.reshape(1, 1025)
                predicted_magnitudes = np.vstack((predicted_magnitudes, prediction))

            predicted_magnitudes = np.array(predicted_magnitudes)
            print("predicted_magnitudes.shape", predicted_magnitudes.shape)

            # griff lim: librosa.core.istft
            # lws: lws_processor.istft (possible problem ...)
            audio = spectrogram2audio(predicted_magnitudes, method=method)
            print("audio.shape", audio.shape)

            all_audio += [audio]
            #all_audio += [dataset.griffin_lim(predicted_magnitudes.T, griffin_iterations)]
            # all_audio += [dataset.lws(predicted_magnitudes.T)]
            # Btw this test shows that the griffin recreation is still pretty bad!
            # maybe replace griffin_lim by LWS?

        return np.array(all_audio)

audio = recreate_samples(x_frames, y_frames, method,idx_start=500)
print("recreated", audio.shape)
scipy.io.wavfile.write("/home/ubuntu/Projects/music_gen_interaction_RTML/TEST2_recreated"+str(method)+".wav", sample_rate, audio[0])

#import librosa
#librosa.output.write_wav('TEST1_recreatedNotShuffledUsedGriffLim.wav', audio[0], sample_rate)
