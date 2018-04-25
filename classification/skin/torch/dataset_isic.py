from torch.utils.data.dataset import Dataset


class ISIC2017Dataset(Dataset):
    def __init__(self):

    def __getitem__(self, index):
        return (img, label)

    def __len__(self):
        return count  # of how many examples(images?) you have

