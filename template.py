import os 

import logging 


#logging string 
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

Project_name='kidney disease classification'


list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{Project_name}/__init__.py",
    f"src/{Project_name}/components/__init__.py",
    f"src/{Project_name}/utils/__init__.py",
    f"src/{Project_name}/config/__init__.py",
    f"src/{Project_name}/config/configuration.py",
    f"src/{Project_name}/pipeline/__init__.py",
    f"src/{Project_name}/entity/__init__.py",
    f"src/{Project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"

]



for filpath in list_of_files:
    filedir,filename = os.path.split(filpath)

    if filedir !="":
       os.makedirs(filedir,exist_ok=True)
       logging.info(f"Creating Diractory ; {filedir} for the: {filename}")
    

    if (not os.path.exists(filpath)) or (os.path.getsize(filpath)==0):
        with open(filpath,'w') as f :
            pass
        logging.info(f"creating file:{filpath}")

    else:
        logging.info(f"{filename} is already exists ")
         