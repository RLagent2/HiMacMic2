REGISTRY = {}

from .basic_controller import BasicMAC
from .agent_controller import AGENTlevel0MAC
from .tsg1_controller import TSGlevel1MAC
from .ns_tsg1_controller import NonSharedTSGlevel1MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["agent_mac"] = AGENTlevel0MAC
REGISTRY["tsg_mac"] = TSGlevel1MAC
REGISTRY["ns_tsg_mac"] = NonSharedTSGlevel1MAC