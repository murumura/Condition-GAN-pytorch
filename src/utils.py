import errno
import os
import shutil
import sys


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def to_np(var: torch.Tensor):
    """Exports torch.Tensor to Numpy array.
    """
    return var.detach().cpu().numpy()

def create_folder(folder_path):
    """Create a folder if it does not exist.
    """
    try:
        os.makedirs(folder_path)
    except OSError as _e:
        if _e.errno != errno.EEXIST:
            raise