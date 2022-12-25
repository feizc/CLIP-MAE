import os 
import json 
from PIL import Image 
import torch.nn.functional as F

import torch 
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets 
from tqdm import tqdm


def zero_shot_classifier(model, classnames, templates, tokenizer, device):
    with torch.no_grad(): 
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")  # tokenize
            texts = texts['input_ids'].to(device)
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device) 
        return zeroshot_weights 


def zero_shot_run(model, classifier, dataloader, device): 
    with torch.no_grad(): 
        top1, top5, n = 0., 0., 0. 
        progress = tqdm(total=len(dataloader), desc='imagenet zero-shot evaluation') 
        for images, targets in dataloader: 
            images = images.to(device)
            targets = targets.to(device) 

            image_features = model.encode_image(images) 
            logits = 100. * image_features @ classifier

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0) 
            progress.set_postfix({"acc-1": top1 / n, "acc-5": top5 / n})
            progress.update()
    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5 



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


class CC3MDataset(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None, is_train=True):
        train_json_path = os.path.join(input_filename, 'train.json')
        with open(train_json_path, 'r', encoding='utf-8') as f: 
            self.data = json.load(f)

        if is_train == True: 
            self.data = self.data[:-50000]
        else:
            self.data = self.data[-50000:]

        self.input_filename = input_filename 

        self.transforms = transforms
        print('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_path = os.path.join(self.input_filename, self.data[idx]['image'][6:])
        images = self.transforms(Image.open(images_path))
        texts = self.tokenize(self.data[idx]['caption'],
                                padding="max_length",
                                max_length=77,
                                truncation=True,
                                return_tensors="pt",)
        return images, texts['input_ids'].squeeze(0)



def get_cc3m_dataset(args, preprocess_fn, is_train, tokenizer=None):
    input_filename = args.train_data 
    assert input_filename
    
    dataset = CC3MDataset(
        input_filename,
        preprocess_fn,
		tokenizer=tokenizer,
        is_train=is_train)
    
    num_samples = len(dataset) 
    print(num_samples)

    if is_train == True:
        # sampler = torch.utils.data.sampler.RandomSampler(dataset)
        sampler = None
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 10,
        shuffle=is_train,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return dataloader 



def get_imagenet_dataset(args, preprocess_fn, is_train=False):
    root = os.path.join(args.imagenet, 'train' if is_train else 'val') 
    dataset = datasets.ImageFolder(root, transform=preprocess_fn) 
    dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                ) 
    print('imagenet data size: ', len(dataset))
    return dataloader 



def reconstruct_loss(target, pred, mask, norm_pix_loss=False): 
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    if norm_pix_loss: 
        mean = target.mean(dim=-1, keep_dim=True) 
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5 

    loss = (pred - target) ** 2 
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

