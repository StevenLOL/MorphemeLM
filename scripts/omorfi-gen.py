#!/usr/bin/env python
import sys
import subprocess

command = 'cd /usr1/home/austinma/git/omorfi && bash /usr1/home/austinma/git/omorfi/src/bash/omorfi-generate.sh'
p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

while True:
  line = sys.stdin.readline().strip()
  if not line:
    break

  p.stdin.write(line + '\n')

  output = p.stdout.readline()
  while output.strip():
    analysis, word, prob = output.split('\t')
    if prob.strip() == 'inf':
      print 'UNK'
    else:
      print word
    output = p.stdout.readline()
