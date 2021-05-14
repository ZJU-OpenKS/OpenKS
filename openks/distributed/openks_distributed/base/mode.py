#   Copyright (c) 2020 Room 525 Research Group, Zhejiang University.
#   All Rights Reserved.


from __future__ import print_function

__all__ = ['Mode']


class Mode:
    """
    There are various mode for fleet, each of them is designed for different model.
    """
    TRANSPILER = 1
    PSLIB = 2
    COLLECTIVE = 3
