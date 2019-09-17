import requests
from timeit import default_timer as timer
import numpy as np
import scipy.misc
from tqdm import tqdm

# = HANDSHAKE =================================================
PORT = "5000"
Handshake_REST_API_URL = "http://localhost:"+PORT+"/handshake"

payload = {"client": "client", "backup_name":"Bob"}
r = requests.post(Handshake_REST_API_URL, files=payload).json()
print("Handshake request data", r)

# = GET AUDIO =================================================
for len_to_test in [32, 64, 128, 256, 512, 1024]:
    times = []
    print("[L=", len_to_test, "]")

    #for k in tqdm(range(10)):
    for k in range(10):
        t_start_request = timer()
        Handshake_GETAUDIO_API_URL = "http://localhost:"+PORT+"/get_audio"
        payload = {"requested_length": str(len_to_test)}
        r = requests.post(Handshake_GETAUDIO_API_URL, files=payload).json()
        #print("Get audio request data", r)

        audio_response = r["audio_response"]
        audio_response = np.asarray(audio_response)

        t_end_request = timer()
        t = t_end_request-t_start_request
        if k > 0: # skip first
            times.append(t)

        if k == 1:
            #print()
            print("     [L=", str(len_to_test).center(5), "] audio_response:", audio_response.shape, "time: ", t, "sec ..", end='', flush=True)
        elif k>1:
            print(".", end='', flush=True)

    times = np.asarray(times)
    print()
    print("  ...[L=", str(len_to_test).center(5), "] time stats:", np.mean(times).round(3) , "+-", np.std(times).round(3) , "sec.")

"""
[local potato pc stats]
[L= 32 ]
     [L=   32  ] audio_response: (36352,) time:  1.3887092260010832 sec ..........
  ...[L=   32  ] time stats: 1.089 +- 0.12 sec.
[L= 64 ]
     [L=   64  ] audio_response: (52736,) time:  1.6142547799991007 sec ..........
  ...[L=   64  ] time stats: 1.877 +- 0.334 sec.
[L= 128 ]
     [L=  128  ] audio_response: (85504,) time:  3.6066677039998467 sec ..........
  ...[L=  128  ] time stats: 3.228 +- 0.256 sec.
[L= 256 ]
     [L=  256  ] audio_response: (151040,) time:  6.081517657001314 sec ..........
  ...[L=  256  ] time stats: 5.99 +- 0.172 sec.
[L= 512 ]
     [L=  512  ] audio_response: (282112,) time:  13.498721015001138 sec ..........
  ...[L=  512  ] time stats: 14.479 +- 2.397 sec.
[L= 1024 ]
     [L=  1024 ] audio_response: (544256,) time:  26.581883702001505 sec ..........
  ...[L=  1024 ] time stats: 23.619 +- 1.78 sec.
"""