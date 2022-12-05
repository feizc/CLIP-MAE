import torch 

from open_clip import CLIPTextCfg, CLIPVisionCfg
from mae_clip import MaeClip 

model = MaeClip(embed_dim=768, vision_cfg=CLIPVisionCfg, text_cfg=CLIPTextCfg)
print(model)


