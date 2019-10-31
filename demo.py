import os
from multiprocessing import Pool

import subprocess
subprocess.run(["killall", "jackd"])
subprocess.Popen(["jackd", "-R", "-d", "alsa", "-r", "44100"])

processes = ('server.py', 'osc_interaction.py', 'client__playbackWithServer.py')

def run_process(process):
    os.system('python {}'.format(process))


pool = Pool(processes=3)
pool.map(run_process, processes)

