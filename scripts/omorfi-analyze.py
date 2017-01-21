#!/usr/bin/env python
import sys
import subprocess

command = 'cd /usr1/home/austinma/git/omorfi && bash /usr1/home/austinma/git/omorfi/src/bash/omorfi-analyse-tokenised.sh'
p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

while True:
  line = sys.stdin.readline().strip()

  if not line:
    break

  for word in line.split():
    p.stdin.write(word + '\n')

    output = p.stdout.readline()
    while output.strip():
      _, analysis, prob = output.strip().split('\t')
      if prob == 'inf':
        print 'UNK'
      else:
        print analysis
      output = p.stdout.readline()
