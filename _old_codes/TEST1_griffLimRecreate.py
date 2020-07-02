audio_data_path = "/home/ubuntu/Projects/music_gen_interaction_RTML/new_audio_samples/dnb/"

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

method="Griff"
#method="LWS"

from audio_handler import AudioHandler
import utils

audio_handler = AudioHandler(griffin_iterations = 60, fft_size=fft_size, window_size=window_size, hop_size=hop_size, sample_rate=sample_rate, sequence_length=sequence_length)
# This uses: librosa.stft and librosa.magphase
dataset = audio_handler.load_dataset(audio_data_path)

print("Dataset:", dataset.x_frames.shape, dataset.y_frames.shape)
dataset.x_frames = np.asarray(dataset.x_frames, dtype=np.float32)
dataset.y_frames = np.asarray(dataset.y_frames, dtype=np.float32)
print("Loaded: dataset.x_frames", dataset.x_frames.shape, "dataset.y_frames", dataset.y_frames.shape)

print("recreating with method", method)

def recreate_samples(x_frames, y_frames, audio_handler, method, idx_start=0, amount_samples=1, sequence_max_length=2000,
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
            audio = audio_handler.spectrogram2audio(predicted_magnitudes, method=method)

            all_audio += [audio]
            #all_audio += [dataset.griffin_lim(predicted_magnitudes.T, griffin_iterations)]
            # all_audio += [dataset.lws(predicted_magnitudes.T)]
            # Btw this test shows that the griffin recreation is still pretty bad!
            # maybe replace griffin_lim by LWS?

        return np.array(all_audio)

audio = recreate_samples(dataset.x_frames, dataset.y_frames, audio_handler, method,idx_start=500)
print("recreated", audio.shape)
scipy.io.wavfile.write("/home/ubuntu/Projects/music_gen_interaction_RTML/TEST1_recreatedNotShuffledUsed"+str(method)+".wav", sample_rate, audio[0])

#import librosa
#librosa.output.write_wav('TEST1_recreatedNotShuffledUsedGriffLim.wav', audio[0], sample_rate)
