import requests
from timeit import default_timer as timer
import numpy as np
import scipy.misc
from tqdm import tqdm
import matplotlib.pyplot as plt
from slugify import slugify

# = HANDSHAKE =================================================
PORT = "5000"
#PORT = "2222"
Handshake_REST_API_URL = "http://localhost:"+PORT+"/handshake"

payload = {"client": "client", "backup_name":"Bob"}
print("trying at ",Handshake_REST_API_URL)
r = requests.post(Handshake_REST_API_URL, files=payload).json()
print("Handshake request data", r)

labels = []
data = []


title = "Local slow potato / Model with 3 lstms / Reconstruct griff. 60 it."
title = "[TMP halfrate] Local slow potato / Model with 3 lstms / Reconstruct griff. 60 it."
#title = "GoogleCloud - K80 gpu, n1-standard-2 / Model with 3 lstms / Reconstruct griff. 60 it."
#title = "GoogleCloud - K80 gpu, n1-standard-2 / Model with 2 lstms / Reconstruct griff. 30 it."
sample_rate = 44100
sample_rate = 22050


repetitions = 10 #10
sequence_to_try = [4, 8, 10, 16, 32]
sequence_to_try = [32, 64, 128, 256, 512, 1024]

# = GET AUDIO =================================================
for len_to_test in sequence_to_try:
    labels.append(len_to_test)

    times_total = []
    times_predict = []
    times_reconstruct = []
    times_communication = []
    print("[L=", len_to_test, "]")

    #for k in tqdm(range(repetitions)):
    for k in range(repetitions):
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
            print("     [L=", str(len_to_test).center(5), "] audio_response:", audio_response.shape, "time: ", t_client_total, "sec ..", end='', flush=True)
        elif k>1:
            print(".", end='', flush=True)

    times_total = np.asarray(times_total)
    times_predict = np.asarray(times_predict)
    times_reconstruct = np.asarray(times_reconstruct)
    times_communication = np.asarray(times_communication)
    seconds_playback = np.max([len(audio_response) / sample_rate])

    print()
    print("  ...[L=", str(len_to_test).center(5), "] time total:", np.mean(times_total).round(3), "+-", np.std(times_total).round(3), "sec. Predict=",np.mean(times_predict).round(3), "Reconstruct=",np.mean(times_reconstruct).round(3), "Communicate=",np.mean(times_communication).round(3), "   / Playback=",seconds_playback.round(3), "sec.")
    data.append([times_predict, times_reconstruct, times_communication, times_total, seconds_playback])
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
     [L=   32  ] audio_response: (36352,) time:  1.9282291459994667 sec ..........
  ...[L=   32  ] time total: 1.891 +- 0.167 sec. Predict= 0.576 Reconstruct= 0.374 Communicate= 0.939    / Playback= 0.824 sec.
[L= 64 ]
     [L=   64  ] audio_response: (52736,) time:  2.628898408999703 sec ..........
  ...[L=   64  ] time total: 2.588 +- 0.111 sec. Predict= 1.1 Reconstruct= 0.518 Communicate= 0.968    / Playback= 1.196 sec.
[L= 128 ]
     [L=  128  ] audio_response: (85504,) time:  3.8227550159999737 sec ..........
  ...[L=  128  ] time total: 4.001 +- 0.157 sec. Predict= 2.119 Reconstruct= 0.786 Communicate= 1.093    / Playback= 1.939 sec.
[L= 256 ]
     [L=  256  ] audio_response: (151040,) time:  6.825211506000414 sec ..........
  ...[L=  256  ] time total: 7.049 +- 0.18 sec. Predict= 4.294 Reconstruct= 1.323 Communicate= 1.426    / Playback= 3.425 sec.
[L= 512 ]
     [L=  512  ] audio_response: (282112,) time:  14.06817415200021 sec ..........
  ...[L=  512  ] time total: 13.71 +- 0.466 sec. Predict= 8.804 Reconstruct= 2.436 Communicate= 2.461    / Playback= 6.397 sec.
[L= 1024 ]
     [L=  1024 ] audio_response: (544256,) time:  26.03507128399997 sec ..........
  ...[L=  1024 ] time total: 26.038 +- 0.779 sec. Predict= 17.901 Reconstruct= 4.874 Communicate= 3.247    / Playback= 12.341 sec.
"""


""" [K80 GPU test & DL image tf 1.14.0 m35, mkl] n1-standard-2 (2 vCPUs, 7.5 GB memory), with 1 K80 GPU
>> python3 server.py -lstm_layers 2 -griffin_iterations 30


"""


# Plotting:


data = np.asarray(data)
times_predict = np.asarray([list(a) for a in data[:,0]])
times_reconstruct = np.asarray([list(a) for a in data[:,1]])
times_communication = np.asarray([list(a) for a in data[:,2]])
times_total = np.asarray([list(a) for a in data[:,3]])
seconds_playback = data[:,4]

print("labels",labels)
print("times_predict", times_predict)
print("np.mean(times_predict,axis=1)", np.mean(times_predict,axis=1))
print("seconds_playback", seconds_playback)

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()

ax1 = ax.bar(x-1.5*width, np.mean(times_predict,axis=1), width, yerr=np.std(times_predict,axis=1), label='times_predict')
ax2 = ax.bar(x-0.5*width, np.mean(times_reconstruct,axis=1), width, yerr=np.std(times_reconstruct,axis=1), label='times_reconstruct')
ax3 = ax.bar(x+0.5*width, np.mean(times_communication,axis=1), width, yerr=np.std(times_communication,axis=1), label='times_communication')
ax4 = ax.bar(x+1.5*width, np.mean(times_total,axis=1), width, yerr=np.std(times_predict,axis=1), label='times_total')
print(x)

ax.hlines(seconds_playback, xmin=x-0.4, xmax=x+0.4, colors='r', linestyles='dashed')
ax.set_xticks(x)
ax.set_ylabel('time (sec)')
ax.set_title(title)
ax.set_xticklabels(labels)
ax.legend()


filename = slugify(title)
plt.savefig("plots/"+filename+".png", dpi=200)
#plt.savefig("plots/"+filename+".pdf", dpi=200)

plt.show()
plt.close()
