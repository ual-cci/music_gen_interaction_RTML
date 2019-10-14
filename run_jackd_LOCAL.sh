
#killall jackd -9 &> /dev/null

#jackd -R -d alsa -d hw:1,0

# local pc settings:
jackd -R -d alsa -r 44100
# important choose the 44100 (defaults to 48k on this pc)