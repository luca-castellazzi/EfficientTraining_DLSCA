import enum


class TargetEnum(enum.Enum):

    """
    Enumeration class containing the supported targets of an attack.

    Attributes:
        - SBO: SBox Output
        - HW: Hamming Weight of the SBox Output
    """

    SBO = 'SBO'
    HW = 'HW'
