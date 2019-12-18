# Interactive Generative Music in real-time with ML
Vitek's interactive music generation project, aiming at real-time speeds and interaction with deep learning!

Ideally we will have a strong PC with good GPU as a server and any machine (eg. my potato laptop) connecting to it. Another version is with having a Google Cloud Virtual Machine as the server (which works, but has availability and slow connectivity issues). Finally it can all run on one machine.

## Install:
Project is work in progress, so this will likely change and grow ...

- `pip install Pillow numpy opencv-python PyWavelets tqdm slugify`
- `pip install -U Flask`
- `pip install lws==1.2.6`
- `pip install librosa tflearn`
- `jackd` (https://jackaudio.org/) and the python client library from https://github.com/spatialaudio/jackclient-python/
  - `sudo apt-get install jack-tools`
  - `pip install JACK-Client --user`
- For the nice UI install PyQt4 (or also can be used with PyQt5 after some adjustments)
- Put your data into `__music_samples` and `__saved_models`:
  - for example: `__music_samples/mehldau/input.wav`
  - so that you have: `__saved_models/modelBest_Mehldau.tfl` (_ps: you can use https://github.com/Louismac/MAGNet to train your model - only remember to use the same settings for the model architecture_)
- Adjust parameter `WAIT_if_qout_larger_div` (in `client__playbackWithServer.py`) according to your PC performance (for slower PC set this value lower - to 1 or 2)

## Start with:

First try running the demo (one file to rule them all) by:
- `python demo.py`

If this doesn't work, follow these steps:

You will have to start the jackd on your pc:
- `jackd -R -d alsa -r 44100` (this command might differ from pc to pc, depending on the soundcard and other setup)

Then start a server (on the same pc, or somewhere else with ssh tunneling), client and interaction tool (now crudely made in python, anything sending the right OSC commands would work):
- (server side) `python3 server.py` (possible to specify some settings here _-lstm_layers 3_ etc ... more to come)
- (client side) `python3 client__playbackWithServer.py` (this will change for sure ...)
- (optionally; client side) `python3 osc_interaction.py` (without this it will send only the default values ... not much interactivity)

## Note:

_This readme might be outdated ... I will try to make my best to follow the code, but probably will return to it only when needed._

## (Note) Low performance PC setup:

I was able to run this (both server+client+jackd) on my potato machine (aka no GPU, only CPU = Intel(R) Core(TM) i5-4300U CPU @ 1.90GHz -> goes turbo to 2.60GHz). Jackd sometimes zombifies. Anything better is much more preferable (for later features, faster reactivity and less zombies).

**client__playbackWithServer.py**
```
SIGNAL_requested_lenght = 64 # bigger batches are a bit faster
WAIT_if_qout_larger_div = 1  # less wasteful waiting time, but also less reactive (longer delay)
cross_len = 32 # shorter client side crossfasing (might cause more audible clicks)
```
**server.py**
```
parser.add_argument('-griffin_iterations', help='iterations to use in griffin reconstruction', default='10') 
# this one is the hardest hit I guess, still sounds reasonable though
```

## Demo

Video with the "proto"-UI: https://www.youtube.com/watch?v=w7Sk7RTVs9U

Audio recording: https://soundcloud.com/previtus/ml-jazz-meanderings-ml-generated-sounds-1/s-DCZbx


[![Interactive Music Generation (10-2019)](https://raw.githubusercontent.com/ual-cci/music_gen_interaction_RTML/master/_illustration_img.png?token=AAIV2RWR3M4IZCAGLU5RQQ26AD72E)](https://www.youtube.com/watch?v=w7Sk7RTVs9U "Interactive Music Generation (10-2019)")

More demos at: https://ual-cci.github.io/vitek/ml_gen_music/report_griff_lim.html?version=61bf4b0 (Full page of generated samples with spectrograms!)