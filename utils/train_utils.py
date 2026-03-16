from utils.file_loader import load_csv_files 


def get_pos_wieghts(): 

  data = load_csv_files()

  pos_values = data[data['label'] == 1].shape[0] 
  neg_values = data[data['label'] == 0].shape[0] 

  return neg_values / (pos_values + neg_values)