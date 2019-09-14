
#killall jackd -9 &> /dev/null

jackd -R -d alsa -d hw:1,0

