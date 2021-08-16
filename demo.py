import os
from multiprocessing import Pool

import subprocess
subprocess.run(["killall", "jackd"])
subprocess.run(["pkill", "-f", "client__playbackWithServer.py"]) # prevent frozen instances!
subprocess.run(["pkill", "-f", "server.py"]) # prevent frozen instances!
subprocess.run(["pkill", "-f", "osc_interaction.py"]) # prevent frozen instances!
subprocess.run(["pkill", "-f", "osc_interaction_pyqt.py"]) # prevent frozen instances!
subprocess.run(["pkill", "-f", "osc_interaction_pyqt_plus_midi.py"]) # prevent frozen instances!
# note: jackd might need
# @audio          -       rtprio          99
# in > /etc/security/limits.conf
# and following:
# sudo usermod -a -G audio [USERNAME]


import pygame
import pygame.midi
pygame.midi.init()
pygame.midi.quit()

# usb sound card on 2,0 (see "aplay -l"; card,device comes from 'card 2: Device [USB Audio Device], device 0: USB Audio [USB Audio]')
#subprocess.Popen(["jackd", "-R", "-d", "alsa", "-d", "hw:2,0", "-r", "44100"]) # start jackd with specific rate 44.1kHz
subprocess.Popen(["jackd", "-R", "-d", "alsa", "-r", "44100"]) # start jackd with specific rate 44.1kHz

#processes = ('server.py', 'osc_interaction.py', 'client__playbackWithServer.py')
processes = ('server.py', 'osc_interaction_pyqt.py', 'client__playbackWithServer.py')
#processes = ('server.py', 'osc_interaction_pyqt_plus_midi.py', 'client__playbackWithServer.py')

def run_process(process):
    os.system('python {}'.format(process))


pool = Pool(processes=3)
pool.map(run_process, processes)

