"""
Convenience facade for the core ForenSight components.
Importing `forensight` re-exports the main classes without needing
to reference individual modules.
"""

from models import (  # noqa: F401
    SampleLoader,
    SequenceMatcher,
    DoubleSampleComparator,
    ParserFactory,
    STRSearcher,
    DNASample,
)
from kernel import KernelMatrix  # noqa: F401
from visualize import (  # noqa: F401
    HIDVisualizer,
    ABIChromatogramVisualizer,
    FSAElectropherogramVisualizer,
    DataBandVisualizer,
)
#from utils import calculate_effort  # noqa: F401
