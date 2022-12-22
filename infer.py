import torch 

from open_clip import CLIPTextCfg, CLIPVisionCfg
from open_clip import MaeCLIP 
from transformers import AutoTokenizer 
from utils import reconstruct_loss 


model = MaeCLIP(embed_dim=768, vision_cfg=CLIPVisionCfg, text_cfg=CLIPTextCfg)
model.load_state_dict(torch.load('./ckpt/0.pt', map_location='cpu'))

tokenizer = AutoTokenizer.from_pretrained('ckpt/tokenizer')
image = torch.randn((1, 3, 224, 224)) 
text = 'a cat on grass'
text_id = tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt",)
text_emb = model.encode_text(text_id['input_ids'], normalize=False).unsqueeze(1) # (1, 768) 

x, mask, ids_restore = model.visual.forward_with_mask(image, mask_ratio=0.7) 
pred = model.decoder(x, ids_restore, text_emb=text_emb)
# print(pred.size(), mask.size()) 

image_patch = model.visual.patchify(image) 
loss = reconstruct_loss(image_patch, pred, mask)
print(loss)







