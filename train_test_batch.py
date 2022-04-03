from torch.utils.data import DataLoader, random_split
from dataset import Quark
import utils


data = Quark(utils.path, utils.name1)
train_size = int(0.9 * len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])

train_batch = DataLoader(
    train_data,
    batch_size=utils.batch_size,
    shuffle=True,
    num_workers=8
)

test_batch = DataLoader(
    test_data,
    batch_size=utils.batch_size,
    num_workers=8
)
