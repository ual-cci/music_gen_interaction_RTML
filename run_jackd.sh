
#killall jackd -9 &> /dev/null

jackd -R -d alsa -d hw:1,0 -r 44100
# test if the soundcard actually runs with 44.1k then!
