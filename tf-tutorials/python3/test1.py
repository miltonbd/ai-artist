#!/usr/bin/env python

# There is no switch statement in python. so use if : elif : else:

from __future__ import print_function 
from __future__ import absolute_import 
from __future__ import division


import sys
from collections import deque



c= 1,2,3

print (type (c))


q= deque(['fef','efewf','wefewf'])

print (q)
 

print (sys.version)

list1= ['1','2','3']


print  ( list1 )


a=int (input('Enter An integer: '))

if a <0:
    print ("Negative")
    
elif a == 0:
    print ("Zero")
    
else:
    print ("Positive Buddy")
    
    