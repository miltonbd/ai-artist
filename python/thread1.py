import threading
import time
all=[]

def count(seconds):
    for second in seconds:
        all.append(second)
        time.sleep(1)


t1=threading.Thread(target=count, args=(range(10),))
t1.start()


t2=threading.Thread(target=count, args=(range(20),))
t2.start()


t3=threading.Thread(target=count, args=(range(30),))
t3.start()

t1.join()
t2.join()
t3.join()

print(len(all))