import torch
import torch.nn as nn
import torch.optim as optim
from Data_Loader import Data_Loaders
class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        # STUDENTS: __init__() must initiatize nn.Module and define your network's
        # custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.fc1 = nn.Linear(6, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        output = self.sigmoid(self.fc3(x))
        return output
        # return output


    def evaluate(self, model, test_loader, loss_function):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in test_loader:
                input, target = data['input'], data['label'].unsqueeze(1)
                output = model(input)
                loss = loss_function(output, target)
                total_loss += loss.item()
        return total_loss / len(test_loader)

def predict(model,input_data):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
        return output

def main():
    model = Action_Conditioned_FF()
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for data in data_loaders.train_loader:
            input, target = data['input'], data['label'].unsqueeze(1)
            # print(f'Input:{input}')
            # print(f'Target:{target}')
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss :{loss.item():.4f}")
    
    test_loss = model.evaluate(model,data_loaders.test_loader,criterion)
    print(f"Test Loss: {test_loss:.4f}")
    torch.save(model.state_dict(),'saved/saved_model.pkl')
if __name__ == '__main__':
    main()


