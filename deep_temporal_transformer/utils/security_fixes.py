"""Security utility functions to prevent common vulnerabilities."""
import os
import re
from typing import Union, List


def validate_path(path: str, allowed_extensions: List[str] = None) -> str:
    """
    Validate and sanitize file paths to prevent security vulnerabilities.
    
    Args:
        path: File path to validate
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Validated and normalized path
        
    Raises:
        ValueError: If path is invalid or potentially dangerous
    """
    if not isinstance(path, str):
        raise ValueError("Path must be a string")
    
    if not path.strip():
        raise ValueError("Path cannot be empty")
    
    # Normalize path
    normalized_path = os.path.normpath(path)
    
    # Check for absolute paths
    try:
        if os.path.isabs(normalized_path):
            raise ValueError("Absolute paths not allowed for security")
    except (OSError, TypeError) as e:
        raise ValueError(f"Path validation failed: {e}")
    
    # Check for directory traversal
    if '..' in normalized_path:
        raise ValueError("Path traversal not allowed")
    
    # Check for null bytes
    if '\x00' in normalized_path:
        raise ValueError("Null bytes not allowed in path")
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'[<>:"|?*]',  # Windows reserved characters
        r'^\.',        # Hidden files starting with dot
        r'~',          # Home directory reference
    ]
    
    try:
        for pattern in suspicious_patterns:
            if re.search(pattern, normalized_path):
                raise ValueError(f"Suspicious pattern detected in path: {pattern}")
    except re.error as e:
        raise ValueError(f"Pattern matching failed: {e}")
    
    # Check file extension if specified
    if allowed_extensions:
        try:
            file_ext = os.path.splitext(normalized_path)[1].lower()
            if file_ext not in [ext.lower() for ext in allowed_extensions]:
                raise ValueError(f"File extension {file_ext} not allowed. Allowed: {allowed_extensions}")
        except (AttributeError, TypeError) as e:
            raise ValueError(f"Extension validation failed: {e}")
    
    return normalized_path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove dangerous characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string")
    
    # Remove or replace dangerous characters
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = 'unnamed_file'
    
    return sanitized


def validate_input_data(data: Union[str, int, float], 
                       data_type: str = "string",
                       max_length: int = 1000) -> bool:
    """
    Validate input data to prevent injection attacks.
    
    Args:
        data: Input data to validate
        data_type: Expected data type
        max_length: Maximum allowed length for strings
        
    Returns:
        True if data is valid
        
    Raises:
        ValueError: If data is invalid
    """
    if data is None:
        raise ValueError("Data cannot be None")
    
    if data_type == "string":
        if not isinstance(data, str):
            raise ValueError("Expected string data")
        
        if len(data) > max_length:
            raise ValueError(f"String too long. Max length: {max_length}")
        
        # Check for SQL injection patterns
        sql_patterns = [
            r"('|(\\')|(;)|(\\;))",
            r"((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
            r"((\%27)|(\'))union",
            r"exec(\s|\+)+(s|x)p\w+",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                raise ValueError("Potential SQL injection detected")
        
        # Check for XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                raise ValueError("Potential XSS attack detected")
    
    elif data_type == "numeric":
        if not isinstance(data, (int, float)):
            raise ValueError("Expected numeric data")
        
        if not (-1e10 <= data <= 1e10):
            raise ValueError("Numeric value out of safe range")
    
    return True