import requests
from timeit import default_timer as timer
import numpy as np
import scipy.misc
from tqdm import tqdm

# = HANDSHAKE =================================================
PORT = "5000"
PORT = "2222"
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
    times_communication = []
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

        t_server_total = r["time_server_total"]
        t_client_total = t_end_request-t_start_request

        t_communication = t_client_total - t_server_total # both ways communication included

        if k > 0: # skip first
            times_total.append(t_client_total)
            times_predict.append(t_predict)
            times_reconstruct.append(t_reconstruct)
            times_communication.append(t_communication)

        if k == 1:
            print("     [L=", str(len_to_test).center(5), "] audio_response:", audio_response.shape, "time: ", t, "sec ..", end='', flush=True)
        elif k>1:
            print(".", end='', flush=True)

    times_total = np.asarray(times_total)
    times_predict = np.asarray(times_predict)
    times_reconstruct = np.asarray(times_reconstruct)
    seconds_playback = np.max([len(audio_response) / 44100])

    print()
    print("  ...[L=", str(len_to_test).center(5), "] time total:", np.mean(times_total).round(3), "+-", np.std(times_total).round(3), "sec. Predict=",np.mean(times_predict).round(3), "Reconstruct=",np.mean(times_reconstruct).round(3), "Communicate=",np.mean(times_communication).round(3), "   / Playback=",seconds_playback.round(3), "sec.")

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


""" [K80 GPU test & DL image tf 1.14.0 m35, mkl] n1-standard-2 (2 vCPUs, 7.5 GB memory), with 1 K80 GPU
[L= 32 ]
     [L=   32  ] audio_response: (36352,) time:  1.7745753650001461 sec ..........
  ...[L=   32  ] time total: 2.42 +- 1.4 sec. Predict= 0.599 Reconstruct= 0.369    / Playback= 0.824 sec.
[L= 64 ]
     [L=   64  ] audio_response: (52736,) time:  2.741011834999881 sec ..........
  ...[L=   64  ] time total: 2.583 +- 0.152 sec. Predict= 1.044 Reconstruct= 0.513    / Playback= 1.196 sec.
[L= 128 ]
     [L=  128  ] audio_response: (85504,) time:  4.072331190000114 sec ..........
  ...[L=  128  ] time total: 4.06 +- 0.158 sec. Predict= 2.107 Reconstruct= 0.778    / Playback= 1.939 sec.
[L= 256 ]
     [L=  256  ] audio_response: (151040,) time:  6.945374451999669 sec ..........
  ...[L=  256  ] time total: 7.253 +- 0.319 sec. Predict= 4.344 Reconstruct= 1.328    / Playback= 3.425 sec.
[L= 512 ]
     [L=  512  ] audio_response: (282112,) time:  13.079125667999506 sec ..........
  ...[L=  512  ] time total: 13.949 +- 0.928 sec. Predict= 8.634 Reconstruct= 2.405    / Playback= 6.397 sec.


"""