import HFT from HFT
import sparse_data from sparse_data
from surprise_sample import run_knn_baseline, run_latent_factor

if __name__ == '__main__':
  data = sparse_data('Video_Games_5.json')
  print("===============================knn====================================")
  run_knn_baseline(data)
  print("===============================latent-factor==========================")
  run_latent_factor(data)
  print("===============================HFT====================================")
  HFT(data)
