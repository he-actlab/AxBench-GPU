#!/usr/bin/python

import sys
import math


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printUsage():
	print "Usage: python qos.py <original file> <nn file>"
	exit(1)
pass;


if(len(sys.argv) != 3):
	printUsage()

origFilename 	= sys.argv[1]
nnFilename		= sys.argv[2]

origLines 		= open(origFilename).readlines()
nnLines			= open(nnFilename).readlines()

sum_delta2 = 0
sum_ref2   = 0
max_delta_ref = 0

for i in range(len(origLines)):

	origLine 	= origLines[i].rstrip()
	nnLine 		= nnLines[i].rstrip()

	origValue 	= float(origLine)
	nnValue 	= float(nnLine)

	delta       = (origValue - nnValue) * (origValue - nnValue)
	ref 		= origValue * origValue + origValue * origValue

	if(math.isnan(delta)):
 		delta = 1.0

	if((delta / ref) > max_delta_ref):
		max_delta_ref = delta / ref

	sum_delta2 += delta
	sum_ref2   += ref

pass

print bcolors.FAIL	+ "*** Error: %1.8f" % (math.sqrt(sum_delta2 / sum_ref2)) + bcolors.ENDC