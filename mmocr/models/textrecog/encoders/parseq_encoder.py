from typing import List
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from mmocr.registry import MODELS
from torch import Tensor
from mmocr.structures import TextRecogDataSample


@MODELS.register_module()
class PARSeqEncoder(VisionTransformer):

    def __init__(self,
                 img_size=[32, 128],
                 patch_size=[4, 8],
                 in_chans=3,
                 embed_dim=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 embed_layer=PatchEmbed
                ):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=False)  # these disable the classifier head``

    def forward(self,
                feat: Tensor,
                data_samples: List[TextRecogDataSample] = None):
        # Return all tokens
        return self.forward_features(feat)