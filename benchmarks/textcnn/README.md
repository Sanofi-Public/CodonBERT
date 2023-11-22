Code for training and testing the TextCNN model. 
By default, the TextCNN model is codon-based and for regression tasks, like 
```
python main.py --data_path ../../data/datamodel_mRFP.csv
```
Build a nucleotide-based model: 
```
python main.py --nuc --data_path ../../data/datamodel_mRFP.csv
```
Point to pre-trained embeddings (check `benchmarks/CodonBERT/extract_embed.py` and `notebooks/benchmark_textcnn.ipynb` to extract embeddings from CodonBERT and Codon2vec, resepctively) : 
```
python main.py --data_path ../../data/datamodel_mRFP.csv --embed_file path_to_embedding_file
```
Train a classification model by specifying the number of classes: 
```
python main.py --data_path ../../data/datamodel_ecoli_sample1000.csv --labels 3 
```
Without specification, the trained model is saved to `./cnn_model.pt` by default. 
Do inference with a trained textcnn model:
```
python main.py --data_path ../../data/datamodel_ecoli_sample1000.csv --labels 3 --predict --snapshot ./cnn_model.pt 
```
