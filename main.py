from pipeline.graph_pipeline import get_train_test_loaders
from pipeline.train_pipeline import TrainClassifier 
from utils.io import save_model
from utils.train_utils import get_pos_wieghts
import matplotlib.pyplot as plt 
from pathlib import Path


if __name__ == '__main__': 

  train_loader, validation_loader = get_train_test_loaders(batch_size = 1024, train_frac= 0.8) 

  model = TrainClassifier(train_loader,
                          validation_loader,
                          model_hidden_dim=64,
                          learning_rate= 0.0001,
                          pos_weight=get_pos_wieghts(), 
                          model_dropout= 0.5) 

  print("Training model...")  
  # implement early stopping:
  train_loss, validation_loss = model.train_model(epochs = 75) 

  print("Model training complete, preparing classification report...")
  print(model.score_model()) 

  save_dir = "./trained_model" 

  # model.save_model(save_dir)
  save_model(model.classifier, save_dir)

  print(f"model saved to: {save_dir}")

  print("saving loss plot") 

  plt.plot(range(len(train_loss)), train_loss, label = 'train loss') 
  plt.plot(range(len(validation_loss)), validation_loss, label = 'validation loss') 
  plt.legend()
  plt.grid(True)
  plt.xlabel("Epochs") 
  plt.ylabel("Loss values")
  plt.savefig(Path(save_dir, "training_plot.png"), dpi = 300)

  print(f"plot saved to: {save_dir}")