try:
    from .alpha_features_core import Alpha191  # dal .pyd compilato
except ImportError:
    raise ImportError(
        "C++ extension not found. "
        "Make sure the package was installed correctly with: pip install alpha-features-core"
    )

from . import alpha191
from . import bridge