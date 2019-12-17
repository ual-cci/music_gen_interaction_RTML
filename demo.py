import os
from multiprocessing import Pool

import subprocess
subprocess.run(["killall", "jackd"])
subprocess.run(["pkill", "-f", "client__playbackWithServer.py"]) # prevent frozen instances!
subprocess.run(["pkill", "-f", "server.py"]) # prevent frozen instances!
#subprocess.run(["pkill", "-f", "osc_interaction.py"]) # prevent frozen instances!
subprocess.run(["pkill", "-f", "osc_interaction_pyqt.py"]) # prevent frozen instances!
subprocess.Popen(["jackd", "-R", "-d", "alsa", "-r", "44100"]) # start jackd with specific rate 44.1kHz

#processes = ('server.py', 'osc_interaction.py', 'client__playbackWithServer.py')
processes = ('server.py', 'osc_interaction_pyqt.py', 'client__playbackWithServer.py')

def run_process(process):
    os.system('python {}'.format(process))


pool = Pool(processes=3)
pool.map(run_process, processes)

