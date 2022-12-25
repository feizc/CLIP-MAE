# Text-Guided Masked Autoencoders for Language-Image Pre-training

Overall, we first mask the image, then feed into the image encoder and text encoder to get image and texr features. Next, the text features is used to guid the image mask reconstruction. Finally, we combine the clip loss and reconstruction loss to optimize the model. 

<p align="center">
     <img src="figures/framework.png" alt="clip-mae framework" width = "400">
     <br/>
     <sub><em>
     Overview of the proposed clip-mae framework.
    </em></sub>
</p>


This repository is based on [openclip](https://github.com/mlfoundations/open_clip) and [mae](https://github.com/facebookresearch/mae).
