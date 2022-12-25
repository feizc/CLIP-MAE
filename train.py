import argparse 
import torch 
from torch import optim 

from transformers import AutoTokenizer
from open_clip import create_transform, CLIPVisionCfg, CLIPTextCfg, ClipLoss
from open_clip import MaeCLIP
from utils import get_cc3m_dataset, reconstruct_loss, get_imagenet_dataset 

from utils import zero_shot_classifier, zero_shot_run
from tqdm import tqdm 
from training import imagenet_classnames, openai_imagenet_template 

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/mayuchen/datasets/research/conceptual_captions',
        help="Path to file(s) with training data",
    )
    parser.add_argument(
        "--imagenet",
        type=str,
        default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/fanmingyuan/cephfs_bk/data/ILSVRC2012',
        help="Path to imagenet data for zero-shot accuracy",
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
    parser.add_argument("--mae_loss", type=bool, default=True)
    parser.add_argument("--debug", type=bool, default=False) 

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

    preprocess_train, preprocess_val = create_transform(image_size=224) 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  
    tokenizer = AutoTokenizer.from_pretrained('ckpt/tokenizer')
    model = MaeCLIP(embed_dim=768, vision_cfg=CLIPVisionCfg(), text_cfg=CLIPTextCfg()) 
    model = model.to(device)

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
    val_loader = get_cc3m_dataset(args, preprocess_val, is_train=False, tokenizer=tokenizer)
    imagenet_loader = get_imagenet_dataset(args, preprocess_val, is_train=False)

    for epoch in range(20): 
        model.train() 
        clip_loss = ClipLoss() 
        loss_cum = .0 
        progress = tqdm(total=len(train_loader), desc='mae-clip training') 
        for i, batch in enumerate(train_loader):  
            images, texts = batch 
            images = images.to(device=device)
            texts = texts.to(device=device)

            x, mask, ids_restore, image_features = model.visual.forward_with_mask(images, mask_ratio=0.5) 
            text_features = model.encode_text(texts, normalize=True)
            pred = model.decoder(x, ids_restore, text_emb=text_features.unsqueeze(1))
            if args.mae_loss == True: 
                rec_loss = reconstruct_loss(model.visual.patchify(images), pred, mask)
                total_loss = rec_loss + clip_loss(image_features, text_features, model.logit_scale.exp()) 
            else:
                total_loss = clip_loss(image_features, text_features, model.logit_scale.exp()) 
            total_loss.backward() 
            optimizer.step() 
            optimizer.zero_grad()
            loss_cum += total_loss.item() 
            progress.set_postfix({"loss": loss_cum / (i + 1)})
            progress.update()
            if args.debug == True:
                break 

        model.eval() 
        with torch.no_grad(): 
            progress = tqdm(total=len(val_loader), desc='mae-clip evaluation') 
            acc_cum = .0
            for i, batch in enumerate(val_loader):
                images, texts = batch 
                images = images.to(device=device)
                texts = texts.to(device=device) 

                image_features, text_features, logit_scale = model(images, texts) 
                logits = torch.matmul(text_features, image_features.t()) * logit_scale 
                pred = torch.argmax(logits, dim=-1) 
                accuracy = torch.eq(pred, torch.arange(len(logits), device=logits.device)).sum() / len(logits)
                acc_cum += accuracy.item() 
                progress.set_postfix({"accuracy": acc_cum / (i + 1)})
                progress.update()
                if args.debug == True:
                    break 
        
        # zero-shot for imagenet accuracy
        print('building zero-shot classifier')
        classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, tokenizer, device)
        acc1, _ = zero_shot_run(model, classifier, imagenet_loader, device)
        print('acc1: ', acc1) 
        
        if args.mae_loss == True:
            print('save modeling')
            torch.save(model.state_dict(), './ckpt/' + str(epoch) + '.pt') 


if __name__ == "__main__":
    main()
