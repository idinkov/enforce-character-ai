# Processors package
from .base_processor import BaseProcessor
from .stage0_to_1 import ProviderImportProcessor
from .stage1_to_2 import DuplicateFilterProcessor
from .stage2_to_3 import UpscaleProcessor
from .stage3_to_4 import SimpleResizeProcessor
from .stage4_to_5 import InpaintingProcessor
from .stage5_to_6 import QualityControlProcessor
from .stage6_to_7 import FinalProcessor
