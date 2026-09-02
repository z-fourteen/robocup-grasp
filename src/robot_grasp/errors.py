class RobotGraspError(Exception):
    """Base exception for expected, user-actionable failures."""


class ValidationError(RobotGraspError):
    """Raised when structured input does not meet the project contract."""


class OptionalDependencyError(RobotGraspError):
    """Raised when an optional external dependency is needed."""
