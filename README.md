# Interactive Generative Music in real-time with ML
Vitek's interactive music generation project, aiming at real-time speeds and interaction with deep learning!

Ideally we will have a strong PC with good GPU as a server and any machine (eg. my potato laptop) connecting to it. Another version is with having a Google Cloud Virtual Machine as the server (which works, but has availability and slow connectivity issues). Finally it can all run on one machine.

## Install:
Project is work in progress, so this will likely change and grow ...

- `pip install Pillow, numpy, opencv-python, pywt`
- `pip install -U Flask`
- `pip install librosa, tflearn`
- `jackd` (https://jackaudio.org/) and the python client library from https://github.com/spatialaudio/jackclient-python/

## Start with:

You will have to start the jackd on your pc:
- `jackd -R -d alsa -r 44100` (this command might differ from pc to pc, depending on the soundcard and other setup)

Then start a server (on the same pc, or somewhere else with ssh tunneling), client and interaction tool (now crudely made in python, anything sending the right OSC commands would work):
- (server side) `python3 server.py` (possible to specify some settings here _-lstm_layers 3_ etc ... more to come)
- (client side) `python3 client__playbackWithServer.py` (this will change for sure ...)
- (optionally; client side) `python3 osc_interaction.py` (without this it will send only the default values ... not much interactivity)

## Note:

_This readme might be outdated ... I will try to make my best to follow the code, but probably will return to it only when needed._
