# CodonBERT

Repository containing data, code and walkthrough for methods in the paper [*CodonBERT: large language models for mRNA design and optimization*](https://www.biorxiv.org/content/10.1101/2023.09.09.556981v1).

### Environment Setup

Dependency management is done via [poetry](https://python-poetry.org/).
```
pip install poetry
poetry install
```
Ensure you have CUDA drivers if you plan on using a GPU.

## CodonBERT

The CodonBERT Pytorch model can be downloaded [here](https://cdn.prod.accelerator.sanofi/llm/CodonBERT.zip). The artifact is under a [license](ARTIFACT_LICENSE.md).
The code and repository are under a [software license](SOFTWARE_LICENSE.md).

Pretraining and finetuning scripts are under [benchmarks/CodonBERT](benchmarks/CodonBERT). 

To extract embeddings from the model, use [extract_embed.py](benchmarks/CodonBERT/extract_embed.py). The dataset [sample.fasta](benchmarks/CodonBERT/data/sample.fasta) is included for reference.

## Dataset
Pre-training dataset are under [benchmarks/CodonBERT/data/pre-train](benchmarks/CodonBERT/data/pre-train), [train_seqs_id_1.csv.zip](benchmarks/CodonBERT/data/pre-train/train_seqs_id_1.csv.zip) and [train_seqs_id_2.csv.zip](benchmarks/CodonBERT/data/pre-train/train_seqs_id_2.csv.zip) list the NCBI IDs for all 10 million sequences for pre-training. [train_samples.csv](benchmarks/CodonBERT/data/pre-train/train_samples.csv) provides a training sample. [eval.csv](benchmarks/CodonBERT/data/pre-train/eval.csv) stores the held-out dataset for evaluation. 

To run finetuning, the `--task` flag must be used. All downstream datasets are under [benchmarks/CodonBERT/data/fine-tune](benchmarks/CodonBERT/data/fine-tune). 
As part of the release, we are sharing an [internal dataset](benchmarks/CodonBERT/data/fine-tune/MLOS.csv). Additionally, the data from other published datasets mentioned in the paper that were used for benchmarking are also included.

- [mRFP_Expression.csv](benchmarks/CodonBERT/data/fine-tune/mRFP_Expression.csv) is from [Revealing determinants of translation efficiency via whole-gene codon randomization and machine learning](https://academic.oup.com/nar/article/51/5/2363/7016452)
- [Fungal_expression.csv](benchmarks/CodonBERT/data/fine-tune/Fungal_expression.csv) is from [Kingdom-Wide Analysis of Fungal Protein-Coding and tRNA Genes Reveals Conserved Patterns of Adaptive Evolution](https://academic.oup.com/mbe/article/39/2/msab372/6513383)
- [E.Coli_proteins.csv](benchmarks/CodonBERT/data/fine-tune/E.Coli_proteins.csv) is from [MPEPE, a predictive approach to improve protein expression in E. coli based on deep learning](https://www.sciencedirect.com/science/article/pii/S2001037022000745)
- [Tc-Riboswitches.csv](benchmarks/CodonBERT/data/fine-tune/Tc-Riboswitches.csv) is from [Tuning the Performance of Synthetic Riboswitches using Machine Learning](https://pubs.acs.org/doi/10.1021/acssynbio.8b00207)
- [mRNA_Stability.csv](benchmarks/CodonBERT/data/fine-tune/mRNA_Stability.csv) are from [iCodon customizes gene expression based on the codon composition](https://www.nature.com/articles/s41598-022-15526-7)
- [CoV_Vaccine_Degradation.csv](benchmarks/CodonBERT/data/fine-tune/CoV_Vaccine_Degradation.csv) is from [Deep learning models for predicting RNA degradation via dual crowdsourcing](https://www.nature.com/articles/s42256-022-00571-8)
  - The average of the deg_Mg_50C values at each nucleotide is treated as the sequence-level target. Deg_Mg_50C has the highest correlation with other labels, including deg_pH10, deg_Mg_pH10, and deg_50C.

## TextCNN
Code for training the TextCNN model is in the [textcnn](benchmarks/textcnn/) directory.
Edit `main.py` to point to the desired embeddings and run `python main.py` to train the model.

## Example Notebooks
The [notebooks](notebooks/) folder contains walkthrough Jupyter notebooks for benchmarking the TFIDF model as well as the TextCNN model with a pre-trained word2vec embedding representation. These use [datamodel_mRFP](datamodel_mRFP.csv) as a test dataset.

## Citations
If you find the model useful in your research, please cite our paper:
```bibtex
@article {Li2023.09.09.556981,
	author = {Sizhen Li and Saeed Moayedpour and Ruijiang Li and Michael Bailey and Saleh Riahi and Milad Miladi and Jacob Miner and Dinghai Zheng and Jun Wang and Akshay Balsubramani and Khang Tran and Minnie Zacharia and Monica Wu and Xiaobo Gu and Ryan Clinton and Carla Asquith and Joseph Skalesk and Lianne Boeglin and Sudha Chivukula and Anusha Dias and Fernando Ulloa Montoya and Vikram Agarwal and Ziv Bar-Joseph and Sven Jager},
	title = {CodonBERT: Large Language Models for mRNA design and optimization},
	elocation-id = {2023.09.09.556981},
	year = {2023},
	doi = {10.1101/2023.09.09.556981},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/09/12/2023.09.09.556981},
	eprint = {https://www.biorxiv.org/content/early/2023/09/12/2023.09.09.556981.full.pdf},
	journal = {bioRxiv}
}
```
