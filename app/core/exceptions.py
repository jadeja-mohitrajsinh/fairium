from fastapi import HTTPException, status

class FairSightException(Exception):
    """Base exception for FairSight application."""
    def __init__(self, detail: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)

class ValidationException(FairSightException):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=status.HTTP_400_BAD_REQUEST)

class ServiceException(FairSightException):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

class NotFoundException(FairSightException):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=status.HTTP_404_NOT_FOUND)
