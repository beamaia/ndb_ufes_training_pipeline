import sys
import logging
import coloredlogs
import os

# add color to logs
coloredlogs.install(level='INFO')
log_file = "logs/log.txt"
logger = logging.getLogger(__name__)
# create console handler and set level to info

if not os.path.exists("logs"):
    os.mkdir("logs")

if os.path.exists(log_file):
    file_exists = True
    suffix = 1

    while file_exists:
        new_file = f"logs/log_{str(suffix).zfill(2)}.txt"
        if os.path.exists(new_file):
            suffix += 1
            continue
        break
    
    log_file = new_file
ch = logging.StreamHandler(sys.stdout) 
fileh = logging.FileHandler(log_file)

# create formatter
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
)

# add formatter to ch
ch.setFormatter(formatter)
fileh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fileh)