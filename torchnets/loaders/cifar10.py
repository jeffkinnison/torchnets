from torchnets.loaders.transforms import Whiten

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop, \
                                   Resize, Normalize


def load_data(path, batch_size, augment, random_seed, valid_size=.1,
              shuffle=True, n_workers=4):
    normalize = transforms.Normalize((0.5, 0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5, 0.5))

    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            Whiten(1e-5)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                download=True, transform=train_transform)

    valid_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                download=True, transform=valid_transform)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                download=True, transform=test_transform)

    num_train = len(train_dataset)
    indices = np.arange(num_train)
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                    batch_size=batch_size, num_workers=num_workers,
                    pin_memory=pin_memory)

    return (train_loader, valid_loader, test_loader)
