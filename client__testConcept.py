import requests
from timeit import default_timer as timer
import numpy as np
import scipy.misc
from tqdm import tqdm

# = HANDSHAKE =================================================
PORT = "5000"
#PORT = "2222"
Handshake_REST_API_URL = "http://localhost:"+PORT+"/handshake"

payload = {"client": "client", "backup_name":"Bob"}
print("trying at ",Handshake_REST_API_URL)
r = requests.post(Handshake_REST_API_URL, files=payload).json()
print("Handshake request data", r)

# = GET AUDIO =================================================
for len_to_test in [32, 64, 128, 256, 512, 1024]:
    times_total = []
    times_predict = []
    times_reconstruct = []
    print("[L=", len_to_test, "]")

    #for k in tqdm(range(10)):
    for k in range(10):
        t_start_request = timer()
        Handshake_GETAUDIO_API_URL = "http://localhost:"+PORT+"/get_audio"
        payload = {"requested_length": str(len_to_test)}
        r = requests.post(Handshake_GETAUDIO_API_URL, files=payload).json()
        #print("Get audio request data", r)

        audio_response = r["audio_response"]
        t_predict = r["time_predict"]
        t_reconstruct = r["time_reconstruct"]
        t_decode = r["time_decode"]

        audio_response = np.asarray(audio_response)

        t_end_request = timer()
        t = t_end_request-t_start_request
        if k > 0: # skip first
            times_total.append(t)
            times_predict.append(t_predict)
            times_reconstruct.append(t_reconstruct)

        if k == 1:
            print("     [L=", str(len_to_test).center(5), "] audio_response:", audio_response.shape, "time: ", t, "sec ..", end='', flush=True)
        elif k>1:
            print(".", end='', flush=True)

    times_total = np.asarray(times_total)
    times_predict = np.asarray(times_predict)
    times_reconstruct = np.asarray(times_reconstruct)
    seconds_playback = np.max([len(audio_response) / 44100])

    print()
    print("  ...[L=", str(len_to_test).center(5), "] time total:", np.mean(times_total).round(3), "+-", np.std(times_total).round(3), "sec. Predict=",np.mean(times_predict).round(3), "Reconstruct=",np.mean(times_reconstruct).round(3), "   / Playback=",seconds_playback.round(3), "sec.")

"""
[local potato pc stats]
[L= 32 ]
     [L=   32  ] audio_response: (36352,) time:  1.3887092260010832 sec ..........
  ...[L=   32  ] time total: 1.041 +- 0.059 sec. Predict= 0.484 Reconstruct= 0.497
[L= 64 ] ~ is roughly 1sec
     [L=   64  ] audio_response: (52736,) time:  1.6142547799991007 sec ..........
  ...[L=   64  ] time total: 1.954 +- 0.364 sec. Predict= 1.073 Reconstruct= 0.77
[L= 128 ] ~ 2 sec
     [L=  128  ] audio_response: (85504,) time:  3.6066677039998467 sec ..........
  ...[L=  128  ] time total: 3.145 +- 0.256 sec. Predict= 1.975 Reconstruct= 1.033
[L= 256 ] ~ 4 sec
     [L=  256  ] audio_response: (151040,) time:  6.081517657001314 sec ..........
  ...[L=  256  ] time stats: 5.99 +- 0.172 sec.
[L= 512 ] ~ 7 sec
     [L=  512  ] audio_response: (282112,) time:  13.498721015001138 sec ..........
  ...[L=  512  ] time stats: 14.479 +- 2.397 sec.
[L= 1024 ] ~ 13 sec
     [L=  1024 ] audio_response: (544256,) time:  26.581883702001505 sec ..........
  ...[L=  1024 ] time stats: 23.619 +- 1.78 sec.
"""


"""
[small CPU instance - 1 cpu, 4GB ram cca] == COMPARABLE ACTUALLY ...
[L= 32 ]
     [L=   32  ] audio_response: (36352,) time:  1.2866049699987343 sec ..........
  ...[L=   32  ] time stats: 1.246 +- 0.044 sec.
[L= 64 ]
     [L=   64  ] audio_response: (52736,) time:  1.9895955409992894 sec ..........
  ...[L=   64  ] time stats: 2.012 +- 0.042 sec.
[L= 128 ]
     [L=  128  ] audio_response: (85504,) time:  3.216515119998803 sec ..........
  ...[L=  128  ] time stats: 3.219 +- 0.041 sec.
[L= 256 ]
     [L=  256  ] audio_response: (151040,) time:  5.398241605002113 sec ..........
  ...[L=  256  ] time stats: 5.391 +- 0.09 sec.
"""


""" ['gpu1cpu2-test1' with GPU] 2 vCPUs, 7.5 GB, 1 K80 GPU



"""