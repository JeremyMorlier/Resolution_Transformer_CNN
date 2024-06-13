import os
import stat

def create_dir(dir) :
    if not os.path.isdir(dir) :
        os.mkdir(dir)
        os.chmod(dir, stat.S_IRWXU | stat.S_IRWXO)
