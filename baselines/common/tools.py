#encoding:utf-8
import json
import fnmatch
import copy
import sys
import os

class Namespace(object):
    def __init__(self, kw):
        self.__dict__.update(kw)

    def todict(self):
        return self.__dict__

def mkdir( dir ):
    # TODO: mkdir reccurence
    if not os.path.exists(dir):
        os.mkdir(dir)
        return True
    return False

def pcname():
    return os.uname().nodename

def ispc(name):
    return pcname().__contains__(name)


def FlagFromFile(filepath):
    filepath += '.cmd'
    try:
        with open(filepath, 'r+') as file:
            cmd = file.read()
            cmd = cmd.replace('\n', '')
            return cmd == '1'
    except:
        os.mknod(filepath)
        return False

def print_refresh(*s):
    sys.stdout.write('\r')
    result = ''
    for i in s:
        result += str(i)
    sys.stdout.write(result)
    sys.stdout.flush()
'''
def JSONFile(filepath, *keys):
    value_update = None
    value_default = None
    filepath += '.json'
    if not os.path.exists(filepath):
        print('re')
        with open(filepath, 'w') as f:
            json.dump( {},f )
    #-- return value
    value = None
    with open(filepath, 'r') as f:
        j = json.load( f )
    if keys:
        value = []
        fill_default = False
        for key in keys:
            if key not in j.keys():#--initialize
                fill_default = True
            value.append(j[key])
        value = tuple(value)
        if len(value) == 1:
            value = value[0]

        #--- fill default
        if fill_default:
            with open(filepath, 'w') as f:
                json.dump( j, f )
    else:
        value = j
    #-- write value_update
    if value_update is not None:
        if len(keys) == 0:#whole json
            assert isinstance(value_update, dict)
            with open(filepath, 'w') as f:
                json.dump(value_update, f)
        elif len(keys)==1:#specific keys
            key = keys[0]
            if j[key] != value_update:
                j_new = copy.deepcopy(j)
                j_new[key] = value_update
                with open(filepath, 'w') as f:
                    json.dump(j_new, f)
        else:
            raise Exception('Not supported for keys update. Please use json update directly')
    return  value
'''
# import time
from timeit import default_timer as timer
def reset_time():
    print_time(reset=True)
__print_time_last = None
def print_time(name=None,reset=False):
    global __print_time_last
    if __print_time_last is None:
        __print_time_last = timer()
    if not reset:
        if name is not None:
            str = f'name:{name}'
        else:
            str = ''
        str += f'time:{timer() - __print_time_last:f} s'
        print(str)
    __print_time_last = timer()


def _get_files_dirs(path_root='', path_rel='', filter_=None, only_sub=True, type='file', dir_end='', sort=None, suffix=None):
    if suffix is not None:
        assert suffix[0] == '.'
        filter_suffix = lambda x: x.endswith(suffix)
        if filter_ is not None:
            filter_t = filter_
            filter_ = lambda x: filter_t(x) and filter_suffix(x)
        else:
            filter_ = filter_suffix
    return _get_files_dirs_entity(path_root, path_rel, filter_, only_sub, type, dir_end, sort)

def _get_files_dirs_entity(path_root='', path_rel='', filter_=None, only_sub=True, type='file', dir_end='', sort=None, suffix=None):
    files = []
    dirs = []
    lists = os.listdir(os.path.join(path_root, path_rel))

    for item in lists:
        item_absolute = os.path.join(path_root, path_rel, item)
        item_rel = os.path.join(path_rel, item)
        if os.path.isfile(item_absolute):
            files.append(item_rel)
        elif os.path.isdir(item_absolute):
            dirs.append(item_rel+dir_end)
    # print(dirs)
    if filter_:
        files = list(filter(filter_, files))
        dirs_search = copy.copy(dirs)
        dirs = list(filter(filter_, dirs))
    else:
        dirs_search = copy.copy(dirs)

    if type =='file':
        #if filter_: files = list(filter( filter_, files )) #fnmatch.filter(files, filter_)
        if not only_sub:
            for dir in dirs_search:
                files += _get_files_dirs_entity(path_root, dir, filter_, only_sub, 'file')
        obj_return = files
    elif type == 'dir':
        #if filter_: dirs = list(filter( filter_, dirs ))#fnmatch.filter(dirs, filter_)
        if not only_sub:
            for dir in dirs_search:
                # print(path_root, ' ', path_rel, ' ', dir)
                dirs += _get_files_dirs_entity(path_root, dir, filter_, only_sub, 'dir', dir_end)
        obj_return = dirs
    else:
        NotImplementedError

    if sort is not None:
        obj_return.sort(reverse=sort)
    return obj_return

def get_files(path_root='', path_rel='', filter_=None, only_sub=True, sort=None, suffix=None):
    '''
    :param path_root:
    :type path_root:
    :param path_rel:
    :type path_rel:
    :param filter_:a function returns true or false. e.g. lamabda filename: filename.__contains__('xxx')
    :type filter_:
    :param only_sub:
    :type only_sub:
    :param sort:
    :type sort:
    :param suffix:
    :type suffix:
    :return:
    :rtype:
    '''
    return _get_files_dirs(path_root,path_rel,filter_,only_sub,'file', sort=sort, suffix=suffix)


def get_dirs(path_root='', path_rel='', filter_=None, only_sub=True, dir_end='', sort=None, suffix=None):
    '''

    :param path_root:
    :type path_root:
    :param path_rel:
    :type path_rel:
    :param filter_:a function returns true or false. e.g. lamabda filename: filename.__contains__('xxx')
    :type filter_:
    :param only_sub:
    :type only_sub:
    :param dir_end:
    :type dir_end:
    :param sort:
    :type sort:
    :param suffix:
    :type suffix:
    :return:
    :rtype:
    '''
    return _get_files_dirs(path_root,path_rel,filter_,only_sub,'dir', dir_end=dir_end, sort=sort, suffix=suffix)


import demjson

def load_args(dir_root, dir_sub='', name_trial='', file_name='arg', args_default=None, return_args_str=False):
    '''

    '''
    if dir_sub != '':
        if name_trial != '':
            name_trial = '_' + name_trial
        path_logger = os.path.join(dir_root, dir_sub + name_trial + '/')
    else:
        if name_trial != '':
            path_logger = os.path.join(dir_root, dir_sub + name_trial + '/')
        else:
            path_logger = dir_root

    if not os.path.exists(path_logger):
        os.mkdir(path_logger)
    file_arg = os.path.join(path_logger, file_name +'.json')
    if not os.path.exists(file_arg):
        if args_default is not None:
            args = args_default
            args = args.replace('=', ':')
            with open(file_arg, 'w') as f:
                f.write(args)
        print('Please initialize args in ' + file_arg)
        #exit()#临时
    with open(file_arg, 'r') as f:
        args_str = f.read()
        args = demjson.decode(args_str)
        args = Namespace(args)
    if return_args_str:
        return args,path_logger,file_arg,args_str
    else:
        return args, path_logger, file_arg


def load_config(filename):
    with open(filename, 'r') as f:
        args_str = f.read()
        args = demjson.decode(args_str)
        args = Namespace(args)
        return args

import pickle
from warnings import warn
def load_vars(filename, catchError=False):
    try:
        with open(filename,'rb') as f:
            return pickle.load( f )
    except Exception as e:
        if catchError:
            warn( f'Load Error! {filename}' )
            return None
        raise e

import time

def save_vars(filename, *vs, disp=False):
    if disp:
        print( f'Write to \n{filename}' )
    with open(filename,'wb') as f:
        if len(vs) == 1:
            pickle.dump(vs[0], f)
        else:
            pickle.dump(vs, f)

def save_np(filename, array):
    import numpy as np
    np.savetxt(filename, array, delimiter=',')

def get_ip(name):
    import netifaces as ni

    ni.ifaddresses(name)
    ip = ni.ifaddresses(name)
    if ni.AF_INET in ip.keys():
        ip = ip[ni.AF_INET][0]['addr']
        return ip
    else:
        return None


def time_now_str():
    return time.strftime('%m/%d|%H:%M:%S', time.localtime())

def time_now_str_filename():
    return time.strftime('%m_%d_%H_%M_%S', time.localtime())


import shutil
import re
def check_safe_path(path, confirm=True, depth=4, name='Modify'):
    # print(f"^({os.environ['HOME']}|/media)(/[^/]+){{3,}}")
    # exit()
    assert depth >= 4
    assert re.match(
        ''.join([ "^(", os.environ['HOME'], "|/media)(/[^/]+){",str(depth-1),",}" ])
        ,path), f'At least operate {depth}-th depth sub-directory!'
    if confirm:
        contents = ''
        if not os.path.isfile( path ):
            dirs = get_dirs(path, dir_end='/')
            files = get_files( path )
            for content in dirs+files:
                contents += f'\n       {content}'
        print(f"{name} path '{path}'! It contains {contents}\n (y or n)?", end='')
        cmd = input()
        if cmd == 'y':
            return True
        else:
            return False
    else:
        return True

def safe_move( src, dst, confirm=True ):
    if check_safe_path(src, confirm, name='Move'):
        shutil.move(src, dst)
        print(f"Moved '{src}' \nto '{dst}'")
        return True
    print(f"Cancel moving file '{src}'")
    return False

def safe_delete(path, confirm=True):
    if check_safe_path( path, confirm, name='Delete' ):
        print(f"Deleted '{path}'")
        shutil.rmtree( path )
        return True
    print('Cancel rm file')
    return False




if __name__ == '__main__':
    dirs = get_dirs('/media/d/tt/b', only_sub=False)
    print(dirs)
    # print(os.environ['HOME'])
    # safe_move('/root/a/b/c/','/root/b/')
    # safe_rm('/media/d/t/tttt')
    exit()
    files = get_files('/media/d/e/python/utxm', suffix='.py', filter_=lambda x: 'e' in x )
    print(files)
    exit()
    JSONFile('a')
    print(JSONFile('a', value_update={'a':1}))
    print(JSONFile('a', keys=('b','c')))
    #print(get_files('/media/d/e/baselines/ppo1/result_Run',filter='*.meta'))



#---plt