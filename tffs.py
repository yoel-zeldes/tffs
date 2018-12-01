import os
import sys
from time import time
import argparse
from functools import reduce
import numpy as np
import tensorflow as tf
import contextlib
from errno import ENOENT
from stat import S_IFDIR, S_IFREG
from fuse import FUSE, FuseOSError, Operations

_NUM_CHARS_IN__NUMPY_VAL = 10
_TENSOR_EVALUATION_NOT_SUPPORTED = -1


def _load_model(model_path):
    """
    Load a tensorflow model from the given path.
    It's assumed the path is either a directory containing a .meta file, or the .meta file itself.
    If there's also a file containing the weights with the same name as the .meta file
    (without the .meta extension), it'll be loaded as well.
    """
    if os.path.isdir(model_path):
        meta_filename = [filename for filename in os.listdir(model_path) if filename.endswith('.meta')]
        assert len(meta_filename) == 1, 'expecting to get a .meta file or a directory containing a .meta file'
        model_path = os.path.join(model_path, meta_filename[0])
    else:
        assert model_path.endswith('.meta'), 'expecting to get a .meta file or a directory containing a .meta file'
    weights_path = model_path[:-len('.meta')]

    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(model_path)
        if os.path.isfile(weights_path):
            session = tf.Session(graph=graph)
            saver.restore(session, weights_path)
        else:
            session = None
    return graph, session


def _reduce_mul(l):
    """
    Multiply the elements in the given list.
    """
    return reduce(lambda x, y: x * y, l, 1)


def _np_array_str_size(shape):
    """
    Calculate len(str(arr)) where arr is some np.array with the given shape, assuming
    each value in arr will occupy exactly _NUM_CHARS_IN__NUMPY_VAL chars in the string.
    """
    num_non_empty_lines = _reduce_mul(shape[:-1])
    num_empty_lines = sum((_reduce_mul(shape[:i])) - 1 for i in range(1, len(shape) - 1))
    res = 0
    # line prefix:
    res += len(shape) * num_non_empty_lines
    # line suffix:
    res += sum(_reduce_mul(shape[:i]) for i in range(1, len(shape))) + 1
    # white space between values in the same row:
    res += (shape[-1] - 1) * num_non_empty_lines
    # new line characters:
    res += num_non_empty_lines + num_empty_lines - 1
    # data:
    res += _reduce_mul(shape) * _NUM_CHARS_IN__NUMPY_VAL
    return res


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """
    Create a context manager that changes the print options of np.arrays.
    """
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def _fixed_val_length(val):
    """
    This function is a formatter that will be used by np.
    Given a numvric value, return a string representation of that
    value that contains exactly _NUM_CHARS_IN__NUMPY_VAL chars.
    """
    val = str(val)
    if len(val) >= _NUM_CHARS_IN__NUMPY_VAL:
        val = val[:_NUM_CHARS_IN__NUMPY_VAL - 1] + '_'
    if len(val) < _NUM_CHARS_IN__NUMPY_VAL:
        val = (' ' * (_NUM_CHARS_IN__NUMPY_VAL - len(val))) + val
    return val


def _create_script_file(script):
    """
    Create the data needed by FUSE to represent a file.
    """
    return dict(
        st_mode=(S_IFREG | 0o555),
        st_nlink=1,
        st_size=len(script),
        st_ctime=time(),
        st_mtime=time(),
        st_atime=time()
    )


def _create_tensor_file(tensor):
    """
    Create the data needed by FUSE to represent a file.
    """
    if tensor.shape.is_fully_defined():
        size = _np_array_str_size(tensor.shape.as_list() or [1])
    else:
        size = _TENSOR_EVALUATION_NOT_SUPPORTED
    return dict(
        st_mode=(S_IFREG | 0o444),
        st_nlink=1,
        st_size=size,
        st_ctime=time(),
        st_mtime=time(),
        st_atime=time(),
        extended_attrs=dict(shape=str(tensor.shape),
                            dtype=tensor.dtype.name,
                            inputs=('\n' + ' ' * len('inputs: ')).join([t.name for t in tensor.op.inputs]),
                            outputs=('\n' + ' ' * len('outputs: ')).join([t.name for t in tensor.op.outputs]))
    )


def _create_dir(t):
    """
    Create the data needed by FUSE to represent a directory.
    """
    return dict(st_mode=(S_IFDIR | 0o444), st_ctime=t, st_mtime=t, st_atime=t, st_nlink=2)


def _read_template_script(template_path, **substitutes):
    """
    Read the given template and substitute placeholders with the given values.
    """
    with open(template_path, 'r') as f:
        script = f.read()
    for placeholder, value in substitutes.items():
        ind = script.find('{{' + placeholder + '}}')
        script = script[:ind] + value + script[ind + len('{{' + placeholder + '}}'):]
    return script


class TfFs(Operations):
    def __init__(self, mount_point, model_path):
        self._graph, self._session = _load_model(model_path)
        self._files = {}
        self._bin_scripts = {}
        self._tensor_values = {}
        now = time()
        self._files['/'] = _create_dir(now)
        self._files['/bin'] = _create_dir(now)
        self._populate_bin(mount_point)

        for op in self._graph.get_operations():
            for tensor in op.outputs:
                next_slash_index = 0
                while next_slash_index >= 0:
                    next_slash_index = tensor.name.find('/', next_slash_index + 1)
                    if next_slash_index >= 0:
                        key = '/' + tensor.name[:next_slash_index]
                        if key not in self._files:
                            self._files[key] = _create_dir(now)
                self._files['/' + tensor.name] = _create_tensor_file(tensor)

    def _populate_bin(self, mount_point):
        """
        Populate the /bin/ folder with scripts.
        Currently there are two scripts:
        - /bin/inputs - print the inputs of a given tensor.
        - /bin/outputs - print the outputs of a given tensor.
        """
        for script_filename, tensor_to_tensors in [('inputs', self._create_tensor_to_inputs),
                                                   ('outputs', self._create_tensor_to_outputs)]:
            script = _read_template_script('print_tensor_dependencies.py',
                                           MOUNT_POINT_PLACEHOLDER="'{}/'".format(mount_point),
                                           TENSOR_TO_TENSORS_PLACEHOLDER=str(tensor_to_tensors()))
            self._bin_scripts['/bin/' + script_filename] = script
            self._files['/bin/' + script_filename] = _create_script_file(script)

    def _get_all_tensors(self):
        """
        Return all tensors found in the graph.
        """
        return [tensor.name for tensor in sum((op.outputs for op in self._graph.get_operations()), [])]

    def _create_tensor_to_inputs(self):
        """
        Create a map from a tensor's name to the names of all its input tensors.
        """
        return {tensor: [input_tensor.name
                         for input_tensor in self._graph.get_tensor_by_name(tensor).op.inputs]
                for tensor in self._get_all_tensors()}

    def _create_tensor_to_outputs(self):
        """
        Create a map from a tensor's name to the names of all its output tensors.
        """
        res = {}
        for tensor, inputs in self._create_tensor_to_inputs().items():
            for input in inputs:
                res.setdefault(input, []).append(tensor)
        return res

    def _eval_tensor_if_needed(self, path):
        """
        Given a path to a tensor file, evaluate the tensor and cache the result in self._tensor_values.
        """
        if self._session is None:
            return None
        if path not in self._tensor_values:
            self._tensor_values[path] = self._session.run(self._graph.get_tensor_by_name(path[1:]))
        return self._tensor_values[path]

    def read(self, path, size, offset, fh):
        """
        FUSE function called to read a file.
        """
        if path.startswith('/bin/'):
            return self._bin_scripts[path][offset:offset + size]

        val = self._eval_tensor_if_needed(path)
        with printoptions(suppress=True,
                          formatter={'all': _fixed_val_length},
                          threshold=sys.maxint,
                          linewidth=sys.maxint):
            return str(val)[offset:offset + size]

    def readdir(self, path, fh):
        """
        FUSE function called to read a directory.
        """
        files = [file for file in self._files.keys() if path in file]
        files = [file[:(file + '//').index('/', len(path) + 1)] for file in files]
        files = set(files)
        files = ['.', '..'] + [x[len(path) + (1 if path[-1] != '/' else 0):] for x in files if x != path]
        return files

    def getattr(self, path, fh=None):
        """
        FUSE function called to get the attributes of a path.
        """
        if path not in self._files:
            raise FuseOSError(ENOENT)

        return self._files[path]

    def listxattr(self, path):
        """
        FUSE function called to get the names of extended attributes of a path.
        """
        return self._files[path].get('extended_attrs', {}).keys()

    def getxattr(self, path, name, position=0):
        """
        FUSE function called to get the value of an extended attribute of a path.
        """
        xattrs = self._files[path].get('extended_attrs', {})
        if name in xattrs:
            return xattrs[name]
        raise FuseOSError(ENOENT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
Map a tensorflow model into Filesystem in Userspace (FUSE).
Usage examples (assuming mounted to ~/tf):
    find ~/tf                               # list all scopes and tensors
    find ~/tf -type f                       # list all tensors
    xattr -l ~/tf/.../tensor                # get attributes of a tensor
    cat ~/tf/.../tensor                     # print the value found in a tensor
    ~/tf/bin/inputs -d 3 ~/tf/.../tensor    # print the inputs to a tensor, recursively
    ~/tf/bin/outputs --no-fs .../tensor     # print the outputs to a tensor,
                                            # without using the mount prefix
''',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mount', dest='mount', default=os.path.expanduser('~/tf'), help='mount point')
    parser.add_argument('--model', dest='model', required=True, help='path to tensorflow model')
    args = parser.parse_args()

    FUSE(TfFs(mount_point=args.mount, model_path=args.model), args.mount, foreground=True)
