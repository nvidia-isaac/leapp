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


class TracingLock:
    """Singleton class to manage global function tracing lock.
    
    This acts as a global mutex to prevent re-entrant tracing across the entire
    process. When _is_active_function_tracing is True, sys.settrace is active
    for a node and no other tracing should be started.
    """
    _instance = None
    
    def __new__(cls):
        """Singleton implementation - only one instance allowed."""
        if cls._instance is None:
            cls._instance = super(TracingLock, cls).__new__(cls)
            cls._instance._is_active_function_tracing = False
        return cls._instance
    
    @property
    def is_active(self) -> bool:
        """Check if function tracing is currently active.
        
        Returns:
            bool: True if sys.settrace is active for a node, False otherwise.
        """
        return self._is_active_function_tracing
    
    def acquire(self):
        """Acquire the tracing lock.
        
        Raises:
            Exception: If tracing lock is already acquired.
        """
        if self._is_active_function_tracing:
            raise Exception("Tracing lock is already acquired")
        self._is_active_function_tracing = True
    
    def release(self):
        """Release the tracing lock."""
        self._is_active_function_tracing = False
    
    def reset(self):
        """Reset the tracing lock to False.
        
        This is useful for error recovery or reset scenarios.
        """
        self._is_active_function_tracing = False

