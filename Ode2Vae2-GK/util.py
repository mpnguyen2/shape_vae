import os
from datetime import datetime

import absl
from absl import logging

LOGS_DIR = './logs'


def setup_logging(label: str):
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    timestamp = datetime.now().strftime(f'{label}-%m-%d-%Y--%H-%M-%S')

    logging.get_absl_handler().use_absl_log_file(timestamp, LOGS_DIR)
