# Text-Guided Masked Autoencoders for Language-Image Pre-training

Overall, we first mask the image, then feed into the image encoder and text encoder to get image and texr features. Next, the text features is used to guid the image mask reconstruction. Finally, we combine the clip loss and reconstruction loss to optimize the model. 

This repository is based on [openclip](https://github.com/mlfoundations/open_clip).
