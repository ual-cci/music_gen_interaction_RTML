audio_data_path = "/media/vitek/Data/Vitek/Projects/2019_LONDON/music generation/small_file/"
#"""
from utils.audio_dataset_generator import AudioDatasetGenerator

sample_rate          = 44100
fft_settings         = [2048, 1024, 512]
fft_size             = fft_settings[0]
window_size          = fft_settings[1]
hop_size             = fft_settings[2]
sequence_length = 40
sample_rate = 44100

dataset = AudioDatasetGenerator(fft_size, window_size, hop_size, sequence_length, sample_rate)

print("loading")
dataset.load(audio_data_path, force=True, prevent_shuffling=True)

import numpy as np
np.save("data/tmp_y_frames_6000NoShuffle.npy", dataset.y_frames[0:6000])

mlkmlmlk
print("recreating")
audio = dataset.recreate_samples(idx_start=500)
np.save("data/recreatedaudiosNoShuffle.npy", audio)

mlkmlmlk
#"""

import numpy as np
audio = np.load("data/recreatedaudiosNoShuffle.npy")

sample_rate = 44100

"""
print("conv to wav")
from IPython.display import Audio
i = 0
a = Audio(audio[0], rate=sample_rate)
"""



import scipy.io.wavfile
scipy.io.wavfile.write("data/feee.wav", sample_rate, audio[0])
# struct.error: ushort format requires 0 <= number <= (0x7fff * 2 + 1)

#import librosa
#librosa.output.write_wav('data/recreatedautio_test.wav', audio[0], sample_rate)
# struct.error: ushort format requires 0 <= number <= (0x7fff * 2 + 1)