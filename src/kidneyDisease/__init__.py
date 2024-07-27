import os 
import sys 
import logging


logging_str = "[%(asctime)s :%(levelname)s: %(module)s:%(message)s]"

log_dir = "loggs"
log_file_path = os.path.join(log_dir,"runing_logs.log")
os.makedirs(log_dir,exist_ok=True)



logging.basicConfig(
    level=logging.INFO,
    formate= logging_str,

    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]


)

logger = logging.getLogger('kidneyDisease')