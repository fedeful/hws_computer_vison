from __future__ import print_function

from os.path import join,abspath,isdir,isfile
from os import getcwd,listdir, chdir


class ProjectFileManager(object):

    def __init__(self, print_mode=False):
        self.print_mode = print_mode

    def get_files_list(self, extensions=['.jpg']):
        files_list = []
        prev_path = getcwd()

        if self.print_mode:
            print(prev_path)

        for f in listdir(prev_path):
            if isdir(f):
                chdir(abspath(f))
                if type(extensions) == list and len(extensions) > 0:
                    for elem in listdir(getcwd()):
                        for i in extensions:
                            if elem.endswith(i):
                                print(abspath(elem))
                                files_list.append(abspath(elem))

            chdir(prev_path)

        return files_list
