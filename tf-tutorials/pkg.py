from __future__ import print_function

from pkg1 import mod1
from pkg1 import b
from pkg1.pkg2

# every directory in python should be a package

if __name__=="__main__":
    b()
    mod1.a()
    