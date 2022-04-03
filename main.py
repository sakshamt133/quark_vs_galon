from train_test_batch import train_batch, test_batch
from model import MyModel
import torch

model = MyModel()
lr = 0.001
epochs = 5
opti = torch.optim.Adam(model.parameters(), lr=lr)
loss = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    for epoch in range(epochs):
        for (data, label) in train_batch:
            y_hat = model(data)
            l = loss(y_hat, label)
            print(f"for epoch {epoch} loss is  {l}")
            l.backward()
            opti.step()
            opti.zero_grad()


