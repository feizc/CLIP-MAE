import os 
import json 
from PIL import Image 

import torch 
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler



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
        #sampler = torch.utils.data.sampler.RandomSampler(dataset)
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

