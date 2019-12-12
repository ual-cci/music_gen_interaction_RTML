# Audio processing follows bits from https://github.com/Louismac/MAGNet
import numpy as np
import librosa
import utils.audio_dataset_generator
import lws

class AudioHandler(object):
    """
    Will handle everything around processing audio.
    Audio representation conversions and reconstructions.

    - raw audio 2 spectrogram
    - spectrogram 2 reconstructed audio

    """

    def __init__(self, griffin_iterations = 60, fft_size=2048, window_size=1024, hop_size=512, sample_rate=44100, sequence_length=40):
        self.griffin_iterations = griffin_iterations

        self.fft_size = fft_size
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.sequence_length = sequence_length

        # Init the LWS:
        lws_L = 5
        self.lws_processor = lws.lws(self.window_size, self.hop_size, L=lws_L, fftsize=self.fft_size, mode="music")


    def spectrogram2audio(self, spectrogram, method="Griff"):
        if method == "Griff":
            audio = self.griffin_lim(spectrogram.T, self.griffin_iterations)
            audio = np.array(audio)

        elif method == "LWS":
            audio = self.lws_reconstruct(spectrogram)
            audio = np.array(audio)


        # sometimes gets error:
        #     if not np.isfinite(y).all():
        #         raise ParameterError('Audio buffer is not finite everywhere')
        # isfinite() -> Test element-wise for finiteness (not infinity or not Not a Number).

        if not np.isfinite(audio).all():
            print("Problem with the signal - it's not finite (contains inf or NaN)")
            print("Signal = ", audio)
            audio = np.nan_to_num(audio)
            print("Attempted hacky fix")

        return audio

    def griffin_lim(self, stftm_matrix, max_iter=100):
        """"Iterative method to 'build' phases for magnitudes."""
        stft_matrix = np.random.random(stftm_matrix.shape)
        y = librosa.core.istft(stft_matrix, self.hop_size, self.window_size)

        if not np.isfinite(y).all():
            print("Problem with the signal - it's not finite (contains inf or NaN)")
            print("Signal = ", y)
            y = np.nan_to_num(y)
            print("Attempted hacky fix")

        for i in range(max_iter):
            if not np.isfinite(y).all():
                print("Problem with the signal - it's not finite (contains inf or NaN), in iteration",i)
                print("Signal = ", y)
                y = np.nan_to_num(y)
                print("Attempted hacky fix inside the iterative method")

            stft_matrix = librosa.core.stft(y, self.fft_size, self.hop_size, self.window_size)
            stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
            y = librosa.core.istft(stft_matrix, self.hop_size, self.window_size)
        return y

    def lws_reconstruct(self, data):
        X0 = data
        ## HAX, doesnt influence it too much tho
        if X0.dtype != "float64":
            X0 = np.asarray(X0, dtype=np.float64)

        X1 = self.lws_processor.run_lws(X0)  # reconstruction from magnitude (in general, one can reconstruct from an initial complex spectrogram)
        print('{:6}: {:5.2f} dB'.format('LWS', self.lws_processor.get_consistency(X1)))

        represented = X1
        # now reconstruct from:
        print("\trepresented shape", represented.shape)

        reconstruction = self.lws_processor.istft(represented)  # where x is a single-channel waveform
        reconstruction = np.asarray(reconstruction)
        print("\toutput reconstruction:", reconstruction.shape)  # (531968,)
        print("\treconstruction data type:", reconstruction.dtype)

        return reconstruction

    def load_dataset(self, dataset_path):
        dataset = utils.audio_dataset_generator.AudioDatasetGenerator(
            fft_size = self.fft_size, window_size = self.window_size, hop_size = self.hop_size,
            sequence_length = self.sequence_length, sample_rate = self.sample_rate)

        try:
            dataset.load(dataset_path, prevent_shuffling=True)
        except:
            dataset = None

        return dataset