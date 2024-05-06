import os, math, logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset


def DataPreprocess(data):
    # Missing Value
    numeric_data = data.select_dtypes(include=['number'])
    data.fillna(numeric_data.mean(), inplace=True)

    # Categorical Variables
    categorical_data = data.select_dtypes(include=['object'])
    numeric_data = data.select_dtypes(include=['number'])

    # Apply one-hot encoding to categorical columns
    if not categorical_data.empty:
        transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_data.columns)
            ],
            remainder='passthrough'  # Pass through numerical columns unchanged
        )
        data_encoded = transformer.fit_transform(data)
    else:
        data_encoded = numeric_data  # No categorical columns, use numeric data as it is

    # Standardize only the numerical columns
    scaler = StandardScaler()
    scaled_numeric_data = scaler.fit_transform(numeric_data)

    # Convert scaled data back to DataFrame
    scaled_numeric_df = pd.DataFrame(scaled_numeric_data, columns=numeric_data.columns)

    # Concatenate one-hot encoded and scaled numerical data
    processed_data = pd.concat([pd.DataFrame(data_encoded), scaled_numeric_df], axis=1)

    return processed_data


def set_logger(log_path):
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        
        self.resnet = models.resnet50()

        self.resnet.conv1 = nn.Conv2d(self.num_classes, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(2048, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.num_classes, 1, 1)
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x


def train(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()
    stop_epochs = 0
    best_loss = math.inf

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for _, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device) # [batch_size, num_classes]
            labels = labels.to(device) # [batch_size, 1]

            optimizer.zero_grad()

            outputs = model(inputs) # [batch_size]
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted_labels = torch.round(outputs)
            correct_predictions += (predicted_labels == labels.unsqueeze(1)).sum().item()
            total_predictions += predicted_labels.size(0)
        
        accuracy = correct_predictions / total_predictions
        
        logging.info(f'[Epochs: {epoch+1}] loss: {running_loss:.3f} accuracy: {accuracy*100:.2f}%')
        
        # Early Stopping Mechanism
        if running_loss > best_loss:
            stop_epochs += 1
            logging.info(f'---------- Stop Epochs: {stop_epochs} ----------')
        elif running_loss <= best_loss:
            torch.save(model.state_dict(), './model/model_%d.pth' % (epoch+1))
            logging.info('---------- Save Best Model ----------')
            best_loss = running_loss
            stop_epochs = 0

        if stop_epochs == 5:
            logging.info('---------- Early Stopping ----------')
            break


def test(model_path, test_loader, criterion, device):
    model = ResNet(20).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            accuracy = (outputs.round() == labels.float().unsqueeze(1)).sum().item() / len(labels)

            running_loss += loss.item()

    logging.info('Test Loss: %.3f' % running_loss)
    logging.info('Test Accuracy: %.2f%' % (accuracy*100))


def latest_model():
    directory = './model/'
    files = os.listdir(directory) if os.path.exists(directory) else os.mkdir(directory)
    model_files = [os.path.join(directory, file) for file in files]
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model


def main():
    set_logger('log.txt')

    data = pd.read_csv('data/train_transaction.csv')
    target = data['isFraud']
    data = data.drop(columns=['isFraud'])
    logging.info('------------- Data Loaded -------------')

    scaled_data = DataPreprocess(data)
    logging.info('---------- Data Preprocessed ----------')

    # Calculate Correlation
    correlation = scaled_data.corrwith(target)
    top_features = correlation.abs().nlargest(150)
    selected_features = scaled_data[top_features.index]
    logging.info('---------- Features Selected ----------')

    # Resample Data
    smote = SMOTE(sampling_strategy='auto',random_state=42)
    X_sampled, y_sampled = smote.fit_resample(selected_features, target)
    logging.info('----------- Data  Resampled -----------')

    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)
    logging.info('------------ Data Splitted ------------')

    X_train_tensor = torch.tensor(x_train.values).float()
    X_test_tensor = torch.tensor(x_test.values).long()
    y_train_tensor = torch.tensor(y_train.values).float()
    y_test_tensor = torch.tensor(y_test.values).long()

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 25600
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = X_train_tensor.shape[1]

    model = ResNet(num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    logging.info('----------- Start  Training -----------')
    train(model, train_loader, criterion, optimizer, device, num_epochs=1000)
    logging.info('---------- Training Finished ----------')

    logging.info('------------ Start Testing ------------')
    test(latest_model(), test_loader, criterion, device)
    #test("./model/model_  70.pth", test_loader, criterion, device)
    logging.info('---------- Testing  Finished ----------')

def demo():
    data = pd.read_csv('data/test.csv')

    scaled_data = DataPreprocess(data)
    print(scaled_data.shape)

    X_test_tensor = torch.tensor(scaled_data.values).long()
    test_data = TensorDataset(X_test_tensor)

    batch_size = 10
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = './model/model_46.pth'
    model = ResNet(150).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # model = torch.load('./model/model_46.pth', map_location=torch.device('cpu'))

    model_outputs = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)  
            outputs = model(inputs)
            model_outputs.append(outputs.cpu().numpy()) 

    model_outputs = np.concatenate(model_outputs, axis=0)

    output_df = pd.DataFrame({'TransactionID': data['TransactionID'], 'isFraud': model_outputs})

    output_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()