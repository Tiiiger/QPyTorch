#!/usr/bin/expect
spawn {pip uninstall qtorch-cpu qtorch-cuda}
expect "?"
send "y\n"
expect "?"
send "y\n"
send "python quant/quant-cpu/setup.py install \n"
send "python quant/quant-cuda/setup.py install \n"