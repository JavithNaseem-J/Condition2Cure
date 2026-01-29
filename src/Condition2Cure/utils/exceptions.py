"""
Custom Exceptions
=================
Simple exception handling for the project.
"""
import sys


class CustomException(Exception):
    """Custom exception with file and line information."""
    
    def __init__(self, message: str, error_detail: sys = None):
        if error_detail:
            _, _, exc_tb = error_detail.exc_info()
            if exc_tb:
                self.file_name = exc_tb.tb_frame.f_code.co_filename
                self.line_number = exc_tb.tb_lineno
                self.message = f"[{self.file_name}:{self.line_number}] {message}"
            else:
                self.message = str(message)
        else:
            self.message = str(message)
        super().__init__(self.message)
    
    def __str__(self):
        return self.message
