from multiprocessing import Manager, Process, Queue, Event
import numpy as np
import time

def multiple_pop(l, n=1):
    # tmp = l.pop(0)
    for i in range(n):
        l.pop(0)


def consumer(l, e):
    while True:
        if len(l) >= 100:
            print('Consumer 100 -- %d' % l[-1])
            multiple_pop(l, 100)
            time.sleep(1)
        if len(l) == 0:
            e.set()
        else:
            e.clear()

def main():
    counter = 0
    with Manager() as manager:
        shared_list = manager.list()
        e = Event()
        # q = Queue()
        p = Process(target=consumer, args=(shared_list, e))
        p.start()
        while counter < 1000:
            shared_list.append(counter) # np.zeros((16, 2048)))
            counter += 1
            time.sleep(0.05)
            if counter % 100 == 0:
                print('Producer 100')

        e.wait()
        p.terminate()


if __name__ == '__main__':
    main()