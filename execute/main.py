import subprocess
import os
import gc

if __name__ == '__main__':
    path = '../models/'
    methods = os.listdir(path)

    for method in methods:
        os.system('cls')
        gc.collect()
        print(method)
        subprocess.run("python ./main.py", shell=True, cwd=path + method)