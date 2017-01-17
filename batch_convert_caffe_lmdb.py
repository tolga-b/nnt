import os
import convert_caffe_lmdb
import argparse
"""
We convert all lmdb databases in a directory
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert all lmdb databases in directory to h5 or npy')
    parser.add_argument('path_to_lmdb', type=str, help='Full path to lmdb directory')
    parser.add_argument('path_to_output', type=str, help='Full path to output directory')
    parser.add_argument('--save_format', type=str, help='h5 or npy',
                        default='h5', choices=['h5', 'npy'])
    args = parser.parse_args()

    if args.save_format == 'npy':
        save_ext = '.npy'
    else:
        save_ext = '.h5'

    path_to_lmdb = os.path.join(args.path_to_lmdb)
    lmdb_dirs = [name for name in os.listdir(path_to_lmdb)
                 if os.path.isdir(os.path.join(path_to_lmdb, name))]

    for d in lmdb_dirs:
        print('#### Starting {}...'.format(d))
        in_path = os.path.join(path_to_lmdb, d)
        out_path = os.path.join(args.path_to_output, d + save_ext)
        convert_caffe_lmdb.main([in_path, out_path])
