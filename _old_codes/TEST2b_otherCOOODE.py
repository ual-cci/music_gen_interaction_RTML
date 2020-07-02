import sys
import traceback
import librosa
import numpy as np
import lws

TEST_INPUT = './data/test_xhat.wav'
TEST_INPUT = "new_audio_samples/dnb/_dnb.wav"


class AudioProcessor:
    """Audio Processing unit transform mag. spectrum and back using LWS
    """
    def __init__(self, sr=22050, n_fft=1024, hop_sz=512, mono=True):
        """"""
        self.sr = sr
        self.n_fft = n_fft
        self.hop_sz = hop_sz
        self.mono = mono

        # initialize lws processor
        # (can be substituted librosa + Griffin-Lim)
        self._lws = lws.lws(n_fft, hop_sz, mode='music')

    def load(self, fn):
        """"""
        return librosa.load(fn, sr=self.sr, mono=self.mono)[0]

    def logamp(self, X):
        """dB-scaler for power spectrum
        X (ndarray): power spectrum
        """
        return librosa.amplitude_to_db(X)

    def inv_logamp(self, X):
        """scaler for power spectrum
        X (ndarray): dB-scale power spectrum
        """
        return librosa.db_to_amplitude(X)

    def forward(self, x):
        """Forward process (signal --> stft)
        x (ndarray) : time-domain audio signal
        """
        return np.abs(self._lws.stft(x)).astype(np.float32)

    def backward(self, X, logamp=True):
        """Backward procss ((log) mag. stft --> signal)
        X (ndarray) : magnitude spectrum
        logamp (bool) : flag for checking input is logamp or not
        """
        if logamp:
            X = self.inv_logamp(X)

        # casting into double_t
        X = X.astype(np.float64)

        if len(X.shape) > 2:
            # only keep last two dimensions
            # (assuming they are for (time step / frequency bins))
            if X.shape[0] == 1:
                surplus_dims = len(X.shape) - 2
                X = np.squeeze(X, axis=range(surplus_dims))
            else:
                assert ValueError(
                    '[Error] only supports one sample at this moment')

        # fill insufficient frequency coeffs with zeros
        n_pos_freqs = int(((self.n_fft / 2.) + 1))
        if X.shape[-1] < n_pos_freqs:
            d = n_pos_freqs - X.shape[-1]
            X = np.concatenate([X, np.zeros((X.shape[-2], d))], axis=-1)

        return np.real(self._lws.istft(self._lws.run_lws(X)))

    def mel(self, X, logamp=True):
        """"""
        M = librosa.feature.melspectrogram(S=X.T, sr=self.sr).T
        if logamp:
            M = self.logamp(M)
        return M

    def blend(self, x, y, snr=10):
        """Blend two signal with target snr
        x, y (ndarray) : vector of time-domain signal.
        snr (int, float) : target snr. assuming x is signal and
                           y is noise
        """
        if len(x) < len(y):
            y = y[:len(x)]
        elif len(y) < len(x):
            x = x[:len(y)]

        if snr == np.inf:
            scaler, prescaler = 0, 1
        elif snr == -np.inf:
            scaler, prescaler = 1, 0
        else:
            power1 = np.sum(x**2)
            power2 = np.sum(y**2)
            scaler = np.sqrt( power1 / (power2 * 10.**(snr/10.)) )
            prescaler = 1

        # blend
        return prescaler * x + scaler * y

    def blend_spec(self, X, Y, snr=10, logamp=False):
        """Blend two spectrograms with target snr
        X, Y (ndarray) : magnitude spectrum. phase is estimated
                         by LWS processor
        snr (int, float) : target snr. assuming X is signal and
                           Y is noise
        """
        x = self.backward(X)
        y = self.backward(Y)
        return self.blend(x, y, snr)


def test_audio_processor():
    """"""
    ap = AudioProcessor()

    # load test input
    X = ap.forward(ap.load(TEST_INPUT))

    # backward test
    try:
        x_ = ap.backward(X, logamp=False)
    except:
        print('Failed backward processing test')
    else:
        print('backward process test success')

    # forwrad
    try:
        X_ = ap.forward(x_)
    except:
        print('Failed forwarrd processing test')
    else:
        print('forward process test success')

    # mel
    try:
        X_ = ap.mel(ap.forward(x_))
        print ( X_.shape )
    except:
        traceback.print_exc(file=sys.stdout)
        print('Failed mel forwarrd processing test')
    else:
        print('mel forward process test success')


    # blend
    try:
        X0 = X
        X1 = X
        print( X0.shape, X1.shape )
        X01 = ap.blend_spec(X0, X1, snr=30, logamp=False)
    except:
        print('Failed blend_spec processing test')
    else:
        print('blend_spec process test success')

    # blend
    try:
        x0 = ap.backward(X, logamp=False)
        x1 = ap.backward(X, logamp=False)
        x01 = ap.blend(x0, x1, snr=30)
        librosa.output.write_wav('TEST2b___REC_test_blend.wav', x01, ap.sr, norm=True)
    except:
        print('Failed blend processing test')
    else:
        print('blend process test success')


if __name__ == "__main__":
    test_audio_processor()