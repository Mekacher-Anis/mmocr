from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer
from mmocr.registry import MODELS
from mmocr.utils.typing import (ConfigType, InitConfigType, OptConfigType)
from typing import Any


@MODELS.register_module()
class PARSeq(EncoderDecoderRecognizer):
    """Implementation of `PARSeq <https://arxiv.org/pdf/2207.06966.pdf>`"""