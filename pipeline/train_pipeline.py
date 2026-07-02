from model.gnn_model import GraphClassifier
import logging
import torch.nn as nn
from torch.optim import Adam
import torch
from sklearn.metrics import f1_score, classification_report


class TrainClassifier:

    def __init__(
        self,
        train_loader,
        validation_loader,
        model_hidden_dim=13,
        learning_rate=0.0001,
        pos_weight=0.5, 
        model_dropout= 0.3):

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        node_attributes_shape = next(iter(train_loader)).x.shape[-1]

        edge_attributes_shape = next(iter(train_loader)).edge_attr.shape[-1]

        self.classifier = GraphClassifier(
            node_attributes_shape=node_attributes_shape,
            edge_attributes_shape=edge_attributes_shape,
            hidden_dim=model_hidden_dim,
            dropout_rate=model_dropout,
        )

        self.optimizer = Adam(self.classifier.parameters(), learning_rate)

    def train_model(self, epochs=30):

        # put the model in train mode
        self.classifier.train()

        train_loss_values = []
        validation_loss_values = []
        for i in range(epochs):

            total_loss = 0
            for batch in self.train_loader:

                self.optimizer.zero_grad()

                predictions = self.classifier(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )

                loss = self.loss_func(predictions, batch.y.float().unsqueeze(dim=1))
                loss.backward()

                total_loss += loss.item()

                self.optimizer.step()

            print(f"epoch {i}: total loss: {total_loss/len(self.train_loader)}")

            train_loss_values.append(total_loss / len(self.train_loader))

            validation_loss_value = self.validate_model()

            validation_loss_values.append(validation_loss_value)

        return train_loss_values, validation_loss_values

    def validate_model(self):

        self.classifier.eval()

        total_validation_loss = 0

        with torch.no_grad():

            for batch in self.validation_loader:

                prediction = self.classifier(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )

                loss = self.loss_func(prediction, batch.y.float().unsqueeze(dim=1))

                total_validation_loss += loss.item()

        print(
            f"Validation run loss: {total_validation_loss/len(self.validation_loader)}"
        )

        return total_validation_loss / len(self.validation_loader)

    def score_model(self):

        self.classifier.eval()

        predictions = []
        true_values = []

        for batch in self.validation_loader:

            prediction = self.classifier.predict(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )

            true_value = batch.y.float().unsqueeze(dim=1)

            # prediction = torch.where(prediction > 0.5, 1, 0)

            predictions.append(prediction)
            true_values.append(true_value)

        predictions = torch.concat(predictions, dim=0).detach().numpy()

        true_values = torch.concat(true_values, dim=0).detach().numpy()

        macro_f1 = f1_score(true_values, predictions, average="macro")

        print(f"macro f1 score: {macro_f1}")
        print(classification_report(true_values, predictions))
        return 