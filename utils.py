import os 
import json 
from PIL import Image 

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler



class CC3MDataset(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None):
        train_json_path = os.path.join(input_filename, 'train.json')
        with open(train_json_path, 'r', encoding='utf-8') as f: 
            self.data = json.load(f)

        self.input_filename = input_filename 

        self.transforms = transforms
        print('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_path = os.path.join(self.input_filename, self.data[idx]['image'][6:])
        images = self.transforms(Image.open(images_path))
        texts = self.tokenize([str(self.captions[idx]['caption'])])[0]
        return images, texts



def get_cc3m_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CC3MDataset(
        input_filename,
        preprocess_fn,
		tokenizer=tokenizer)
    num_samples = len(dataset) 
    print(num_samples)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return dataloader 

