import numpy as np
import time
import timeit


data = (np.random.random([300,300,300])*1000).astype(np.int16)
data2 = []
ta = []
tb = []

for j in range (0,100):
    t0 = time.time()
    for i in range(1,3000):
        data2 = data[10,:,:]
    t1 = time.time()
    ta.append(t1-t0)
    print 'cas A: ', t1 - t0





    data2 = []
    t0 = time.time()
    for i in range(1,3000):
        data2 = data[:,:,10]
    t1 = time.time()

    tb.append(t1-t0)
    print 'cas B: ', t1 - t0

print 'sum A: ', np.sum(ta)
print 'sum B: ', np.sum(tb)

#import pdb; pdb.set_trace()


#print 'A ', timeit.timeit('data2=data[:,:,1]', number=1000)
#print 'B ', timeit.timeit('data2=data[1,:,:]', number=1000)

