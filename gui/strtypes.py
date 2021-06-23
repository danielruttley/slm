import time

#### custom logging functions
def warning(msg='',exception=None):
    print('\033[33m' + '####\tWARNING\t' + time.strftime('%d.%m.%Y\t%H:%M:%S'))
    if exception is None:
        print('\t' + msg + '\n', '\033[m')
    else:
        print('\t' + msg + '\n\t'+str(exception)+'\n', '\033[m')

def error(msg='',exception=None):
    print('\033[31m' + '####\tERROR\t' + time.strftime('%d.%m.%Y\t%H:%M:%S'))
    if exception is None:
        print('\t' + msg + '\n', '\033[m')
    else:
        print('\t' + msg + '\n\t'+str(exception)+'\n', '\033[m')

def info(msg='',exception=None):
    print('\033[36m' + '####\tINFO\t' + time.strftime('%d.%m.%Y\t%H:%M:%S'))
    if exception is None:
        print('\t' + msg + '\n', '\033[m')
    else:
        print('\t' + msg + '\n\t'+str(exception)+'\n', '\033[m')
