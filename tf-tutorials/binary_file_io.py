import ctypes
import mmap
import platform


print platform.architecture()[0]
print ctypes.sizeof(ctypes.c_voidp)
#It'll be 4 for 32 bit or 8 for 64 bit.