import requests
from timeit import default_timer as timer
import numpy as np
import scipy.misc

# = HANDSHAKE =================================================
PORT = "5000"
Handshake_REST_API_URL = "http://localhost:"+PORT+"/handshake"

payload = {"client": "client", "backup_name":"Bob"}
r = requests.post(Handshake_REST_API_URL, files=payload).json()
print("Handshake request data", r)

# = GET AUDIO =================================================
while True:
    t_start_request = timer()
    Handshake_GETAUDIO_API_URL = "http://localhost:"+PORT+"/get_audio"
    payload = {}
    r = requests.post(Handshake_GETAUDIO_API_URL, files=payload).json()
    #print("Get audio request data", r)

    audio_response = r["audio_response"]
    audio_response = np.asarray(audio_response)

    t_end_request = timer()
    t = t_end_request-t_start_request
    print("audio_response:", audio_response.shape, "time: ", t)