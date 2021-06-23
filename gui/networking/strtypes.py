"""PyDex types
Stefan Spence 13/04/20

Collection of functions for converting types
"""
import re
from distutils.util import strtobool
import time

def BOOL(x):
    """Fix the conversion from string to Boolean.
    Any string with nonzero length evaluates as true 
    e.g. bool('False') is True. So we need strtobool."""
    try: return strtobool(x)
    except (AttributeError, ValueError): return bool(x)

# I later realised that eval() can replace these functions
def strlist(text):
    """Convert a string of a list of strings back into
    a list of strings."""
    return list(text[1:-1].replace("'","").split(', '))

def intstrlist(text):
    """Convert a string of a list of ints back into a list:
    (str) '[1, 2, 3]' -> (list) [1,2,3]"""
    try:
        return list(map(int, text[1:-1].split(',')))
    except ValueError: return []

def listlist(text):
    """Convert a string of nested lists into a
    list of lists."""
    return list(map(intstrlist, re.findall('\[[\d\s,]*\]', text)))


#### custom logging functions
def warning(msg=''):
    print('\033[33m' + '####\tWARNING\t' + time.strftime('%d.%m.%Y\t%H:%M:%S'))
    print('\t' + msg + '\n', '\033[m')

def error(msg=''):
    print('\033[31m' + '####\tERROR\t' + time.strftime('%d.%m.%Y\t%H:%M:%S'))
    print('\t' + msg + '\n', '\033[m')

def info(msg=''):
    print('\033[36m' + '####\tINFO\t' + time.strftime('%d.%m.%Y\t%H:%M:%S'))
    print('\t' + msg + '\n', '\033[m')