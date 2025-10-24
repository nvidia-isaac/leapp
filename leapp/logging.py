#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
from datetime import datetime
import os

# Define custom log level for SECTION (between INFO and WARNING)
SECTION = 25
logging.addLevelName(SECTION, 'SECTION')


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes"""

    # ANSI color codes (matching export_manager.py style)
    COLORS = {
        'DEBUG': '',              # White (no color)
        'INFO': '',               # White
        'SECTION': '\033[1;4;97m',     # Bold white (for section headers)
        'WARNING': '\033[1;33m',  # Bold yellow (matching export_manager.py)
        'ERROR': '\033[91m',      # Bright red
        'CRITICAL': '\033[1;31m',  # Bold red
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to entire message
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        if log_color:
            record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
            record.msg = f"{log_color}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)


class LeappLogger:

    def __init__(self, f):
        self.logger = logging.getLogger('leapp')
        self.logger.setLevel(logging.DEBUG)  # Capture everything
        self.initialized = False

    def configure_logger(self, savepath, verbose):
        self.logger.handlers.clear()

        # log file handler
        if os.path.exists(savepath):
            filepath = os.path.join(savepath, 'log.txt')
            if os.path.exists(filepath):
                filepath = os.path.join(
                    savepath, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            self.file_handler = logging.FileHandler(filepath, mode='a')
            self.file_handler.setLevel(logging.DEBUG)
            # Plain format for file (with timestamp, no colors)
            file_formatter = logging.Formatter(
                '[%(levelname)s]: %(message)s')
            self.file_handler.setFormatter(file_formatter)
            self.logger.addHandler(self.file_handler)
        else:
            raise FileNotFoundError(f"File {filepath} does not exist")

        # log console handler:
        self.console_handler = logging.StreamHandler()
        self.set_verbose(verbose)
        # Use simple format for console (no timestamp, just message)
        self.console_handler.setFormatter(
            ColoredFormatter('[%(levelname)s]: %(message)s'))
        self.logger.addHandler(self.console_handler)

        self.initialized = True

    def set_verbose(self, verbose):
        # When verbose, show everything (DEBUG and above)
        self.console_handler.setLevel(
            logging.DEBUG if verbose else logging.WARNING)

    def debug(self, msg):
        """Log debug message (file only, not console even if verbose)."""
        if self.initialized:
            self.logger.debug(msg)

    def info(self, msg):
        """Log info message (file always, console if verbose)."""
        if self.initialized:
            self.logger.info(msg)

    def section(self, msg):
        """Log section message (file always, console always) - bold in console."""
        if self.initialized:
            # Add blank lines around section for visual separation
            self.logger.log(SECTION, f"{msg}\n")

    def warning(self, msg):
        """Log warning message (file always, console always)."""
        if self.initialized:
            self.logger.warning(msg)

    def error(self, msg):
        """Log error message (file always, console always)."""
        if self.initialized:
            self.logger.error(msg)
