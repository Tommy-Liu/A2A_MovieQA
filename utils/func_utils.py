import os
import re
import shutil


def bb(v):
    if isinstance(v, float):
        return '%.2E' % v
    elif isinstance(v, str):
        return v
    else:
        return repr(v)


def block_print(s, ch='='):
    if not isinstance(s, list):
        s = [s]
    print(ch * (max([len(e) for e in s]) + 5))
    print('\n'.join(s))
    print(ch * (max([len(e) for e in s]) + 5))


def safe_remove(f):
    if os.path.exists(f):
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)


def imdb_key(base_name):
    return base_name.split('.')[0]


def clean_token(l):
    """Clean up Subrip tags."""
    return re.sub(r'<.+?>', '', l)


def make_dirs(d):
    """If the directory dose not exist, make one."""
    if not os.path.exists(d):
        os.makedirs(d)


# Wrapped function
def basename_wo_ext(p):
    """Get the base name of path p without extension"""
    base_name = basename(p)
    base_name = os.path.splitext(base_name)[0]
    return base_name


# Fuck os.path.basename. I wrote my own version.
def basename(p):
    """Get the subdirectory or file name in the last position of the path p."""
    pos = -1
    if p.split('/')[pos] == '':
        pos = -2
    return p.split('/')[pos]


def is_in(a, b):
    """Is a a subset of b ?"""
    return set(a).issubset(set(b))


def intersect(a, b):
    return not set(a).isdisjoint(set(b))
