from __future__ import division

import multiprocessing
from multiprocessing import JoinableQueue, Manager
from timeit import default_timer
import external.caffe_pb2 as caffe_pb2
import lmdb
import numpy as np
import os
import sys
import argparse
from utils import save_h5
"""
We are going to implement a parallel reader to read large lmdb database and load it into memory
Depends on the protobuf definition from caffe (caffe_pb2.py)
#TODO add on the fly functionality for generator like usage
"""


class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_dict):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_dict = result_dict
        self.datum = caffe_pb2.Datum()

    def run(self):
        # proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                # print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            # print '%s: %s' % (proc_name, next_task)
            ind, feat = next_task  # task is tuple
            self.datum.ParseFromString(feat)
            self.result_dict[ind] = np.array(self.datum.float_data, dtype=np.float32)
            self.task_queue.task_done()
        return


def test_lmdb_read_speed(path_to_features, read_count=50000):
    """ Test how fast we can read from lmdb (no decoding) single thread
    """
    env = lmdb.open(os.path.join(path_to_features), readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        cnt = 0
        start = default_timer()
        for key, val in cursor:
            if cnt == read_count:
                break
            cnt += 1
        elapsed = default_timer() - start
    print 'Read {} in {} seconds, {} files/s'.format(cnt, elapsed, cnt / elapsed)
    return cnt / elapsed


def test_lmdb_decode_speed(path_to_features, read_count=5000):
    """ Test how fast we can decode single thread
    """
    env = lmdb.open(os.path.join(path_to_features), readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        cnt = 0
        feats = []
        for key, val in cursor:
            feats.append(val)
            if cnt == read_count:
                break
            cnt += 1
    datum = caffe_pb2.Datum()
    start = default_timer()
    for feat in feats:
        datum.ParseFromString(feat)
        out = np.array(datum.float_data, dtype=np.float32)
    elapsed = default_timer() - start
    print 'Decoded {} in {}, {} decodes/s'.format(cnt, elapsed, cnt / elapsed)
    return cnt / elapsed


def main(raw_args):
    # set up parser
    parser = argparse.ArgumentParser(description='Convert caffe lmdb to h5 or npy file')
    parser.add_argument('path_to_lmdb', type=str, help='Full path to lmdb database')
    parser.add_argument('path_to_output', type=str, help='Full path to output file')
    parser.add_argument('--num_consumers', type=int, default=0,
                        help='Number of cores to use, default is the available number of cores')
    parser.add_argument('--verbosity', type=int, default=1000,
                        help='Print every verbosity conversions')
    args = parser.parse_args(raw_args)
    # print(args)

    # tasks is fed by single sequential reader
    tasks = JoinableQueue(maxsize=500)

    # results is filled by multiple writers with features
    manager = Manager()
    results = manager.dict()

    # start consumers
    if args.num_consumers == 0:
        num_consumers = multiprocessing.cpu_count()
    else:
        num_consumers = args.num_consumers
    print('Creating {} consumers'.format(num_consumers))
    consumers = [Consumer(tasks, results)
                 for i in xrange(num_consumers)]
    for w in consumers:
        w.start()

    # enqueue jobs
    env = lmdb.open(os.path.join(args.path_to_lmdb), readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        cnt = 0
        start = default_timer()
        for key, val in cursor:
            tasks.put((cnt, val))
            if not cnt % args.verbosity:
                print '{},'.format(cnt),
                sys.stdout.flush()
            cnt += 1

    # add a poison pill for each consumer
    for i in xrange(num_consumers):
        tasks.put(None)

    # wait for all of the tasks to finish
    tasks.join()

    elapsed = default_timer() - start
    # convert and sort results in the end w.r.t processing order
    if cnt > 0:
        print('\nConverted {} in {}, {} conversions/s'.format(cnt, elapsed, cnt / elapsed))
        print('with shape {}'.format(results[0].shape[0]))
        results_sorted = np.zeros((cnt, results[0].shape[0]), dtype=np.float32)
        for i in xrange(cnt):
            results_sorted[i] = results[i]

        if args.path_to_output[-4:] == '.npy':
            np.save(args.path_to_output, results_sorted)
        else:
            save_h5(args.path_to_output, results_sorted)
    print('Done.')


if __name__ == '__main__':
    main(sys.argv[1:])
