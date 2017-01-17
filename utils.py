# Utility functions that only depend on core python libraries
import numpy as np
import h5py
import os


def topk(gt, pred, k=5):
    """
    Top k error
    """
    assert(isinstance(gt[0],(int,float)) or len(gt[0])==1)
    gt = np.array(gt, dtype=int)[:,np.newaxis]
    pred = np.array(pred, dtype=int)
    pred = pred[:,:np.min([k,pred.shape[1]])]
    gt = np.repeat(gt,pred.shape[1],axis=1)
    err = np.min(gt!=pred, axis=1)
    return err


def topk_batch(gt, pred, k):
    # we should not receive multiple gt labels
    assert(isinstance(gt[0],(int,float)))
    err = [g not in p[:k] for g, p in zip(gt,pred)]
    return np.array(err)


def save_h5(save_path, input_array):
    data_name, data_ext = os.path.splitext(save_path)
    if data_ext != '.h5':
        save_path += '.h5'
    h5f = h5py.File(save_path, 'w')
    print('Saving to {}'.format(save_path))
    h5f.create_dataset(data_name, data=input_array)
    h5f.close()


def load_h5(load_path):
    h5f = h5py.File(load_path, 'r')
    b = h5f['dataset_1'][:]
    h5f.close()
    return b
