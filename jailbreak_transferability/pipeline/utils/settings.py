import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import yaml

def get_main_working_directory(name):
    path_base = os.getcwd()
    while path_base and not path_base.endswith(name):
        path_base = os.path.dirname(path_base)
    assert path_base, 'Could not find current directory'
    return path_base
    
class Settings():
  
    path_cwd=get_main_working_directory('jbb_transfer_code')
    __base = {'PATH_BASE_DIR': path_cwd}
    
    def __init__(self):
        self.models = _get_config_(path=os.path.join(self.__base['PATH_BASE_DIR'],'jailbreak_transferability/pipeline/config_paths.yaml'))
        
def _get_config_(path :str):
    with open(path,'r') as file:
        config = yaml.safe_load(file)
    return config






