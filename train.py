import argparse 
import torch 
from torch import optim 

from transformers import AutoTokenizer
from open_clip import create_transform, CLIPVisionCfg, CLIPTextCfg, ClipLoss
from open_clip import MaeCLIP
from utils import get_cc3m_dataset 
from tqdm import tqdm 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/mayuchen/datasets/research/conceptual_captions',
        help="Path to file(s) with training data",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.98, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    args = parser.parse_args()
    return args 


def main(): 
    args = parse_args() 
    print(args) 

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    preprocess_train, _ = create_transform(image_size=224) 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  
    tokenizer = AutoTokenizer.from_pretrained('ckpt/tokenizer')
    model = MaeCLIP(embed_dim=768, vision_cfg=CLIPVisionCfg(), text_cfg=CLIPTextCfg()) 

    # create optimizer and scaler
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    train_loader = get_cc3m_dataset(args, preprocess_train, is_train=True, tokenizer=tokenizer) 

    for epoch in range(5): 
        model.train() 
        loss = ClipLoss() 
        loss_cum = .0 
        progress = tqdm(total=len(train_loader), desc='clip training') 
        for i, batch in enumerate(train_loader):  
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale) 
            total_loss.backward() 
            optimizer.step() 
            optimizer.zero_grad()
            loss_cum += total_loss.item() 
            progress.set_postfix({"loss": loss_cum / (i + 1)})


if __name__ == "__main__":
    main()
