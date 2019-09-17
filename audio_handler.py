# Audio processing follows bits from https://github.com/Louismac/MAGNet
import numpy as np
import librosa

class AudioHandler(object):
    """
    Will handle everything around processing audio.
    Audio representation conversions and reconstructions.

    - raw audio 2 spectrogram
    - spectrogram 2 reconstructed audio

    """

    def __init__(self, fft_size=2048, window_size=1024, hop_size=512, sample_rate=44100):
        self.griffin_iterations = 60

        self.fft_size = fft_size
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

    def spectrogram2audio(self, spectrogram):
        audio = self.griffin_lim(spectrogram.T, self.griffin_iterations)
        audio = np.array(audio)
        return audio

    def griffin_lim(self, stftm_matrix, max_iter=100):
        """"Iterative method to 'build' phases for magnitudes."""
        stft_matrix = np.random.random(stftm_matrix.shape)
        y = librosa.core.istft(stft_matrix, self.hop_size, self.window_size)
        for i in range(max_iter):
            stft_matrix = librosa.core.stft(y, self.fft_size, self.hop_size, self.window_size)
            stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
            y = librosa.core.istft(stft_matrix, self.hop_size, self.window_size)
        return y