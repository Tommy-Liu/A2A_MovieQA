import os
import re


def block_print(s, ch='='):
    print(ch * (max([len(e) for e in s]) + 5))
    print('\n'.join(s))
    print(ch * (max([len(e) for e in s]) + 5))


def exist_then_remove(f):
    if os.path.exists(f):
        os.remove(f)


def get_imdb_key(base_name):
    return base_name.split('.')[0]


def clean_token(l):
    """Clean up Subrip tags."""
    return re.sub(r'<.+?>', '', l)


def exist_make_dirs(d):
    """If the directory dose not exist, make one."""
    if not os.path.exists(d):
        os.makedirs(d)


# Wrapped function
def get_base_name_without_ext(p):
    """Get the base name of path p without extension"""
    base_name = get_base_name(p)
    base_name = os.path.splitext(base_name)[0]
    return base_name


# Fuck os.path.basename. I wrote my own version.
def get_base_name(p):
    """Get the subdirectory or file name in the last position of the path p."""
    pos = -1
    if p.split('/')[pos] == '':
        pos = -2
    return p.split('/')[pos]


def is_in(a, b):
    """Is a a subset of b ?"""
    return set(a).issubset(set(b))
