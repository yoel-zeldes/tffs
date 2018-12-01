#!/usr/bin/python

import os
import sys
import argparse

# these placeholders will be substituted by tffs.py with real values
_MOUNT_POINT = {{MOUNT_POINT_PLACEHOLDER}}


# consts used for pretty printing
_DASH = u'\u2500' * 2 + ' '
_HAS_CHILDREN_MIDDLE = u'\u251c' + _DASH
_HAS_CHILDREN_LAST = u'\u2514' + _DASH
_VERTICAL = '|'


def print_dependencies(dependencies, depth, use_fs, prefix = ''):
    """
    Recursively print the given dependencies in a tree format.
    """
    if depth < 1 or dependencies is None:
        return
    for i, dependency in enumerate(dependencies):
        is_last = i < len(dependencies) - 1
        print prefix + (_HAS_CHILDREN_MIDDLE if is_last else _HAS_CHILDREN_LAST) + \
              (_MOUNT_POINT if use_fs else '') + dependency
        print_dependencies(dependencies=_TENSOR_TO_DEPENDENCIES.get(dependency),
                           depth=depth - 1,
                           use_fs=use_fs,
                           prefix=prefix + (_VERTICAL + '   ' if is_last else '    '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="print a tensor's dependencies",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('tensor')
    parser.add_argument('--depth', '-d', dest='depth', default=1, type=int, help='depth to go to')
    parser.add_argument('--no-fs', '-n', dest='no_fs', action='store_true',
                        help='use real tensors names. Otherwise, treat tensor names '
                             'as if they were the names of the files in the file system')
    args = parser.parse_args()

    tensor = args.tensor
    use_fs = not args.no_fs
    if use_fs:
        tensor = os.path.abspath(os.path.expanduser(tensor))
        assert os.path.isfile(tensor), 'given tensor must correspond to a tensor file found in the file system. ' \
                                       'Use --no-fs (-n) if you want to use real tensor names.'
    dependencies = _TENSOR_TO_DEPENDENCIES.get(tensor[len(_MOUNT_POINT) if use_fs else 0:])
    if not dependencies:
        print 'tensor not found'
        sys.exit(1)
    print tensor
    print_dependencies(dependencies=dependencies, depth=args.depth, use_fs=use_fs)
