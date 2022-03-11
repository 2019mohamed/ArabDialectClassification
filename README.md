# ArabDialectClassification

Clean dataset for training:-
https://drive.google.com/file/d/1j7dhaCxcZh4-ZF5EuX9XJUJZqduGOOf1/view?usp=sharing

codes file contains data collection, preprocessing, vectorizer_reducer, and API for collecting data from API, cleaning data, embedding texts to vectors into space(i.e., Euclidean space) with preserving some properties such as semantics and API with flask but, unfortunately, my tiny laptop cannot load models and so on..., respectively.

some_saved_models file contains some of the saved trained models such as PA and MLP...

1- APIs.ipynb : testing api(i.e., flask api) using python-requests.

2- MLP_DL.ipynb : training using MLP and api.

3- SVM_MLP_Boostrap.ipynb : training using SVM and MLP reducer with bootstrapping.

4- SVM_TruncatedSVD.ipynb : training using SVM and truncatedSVD reducer.

5- Transformer_finetune.ipynb : Finetuning BERT model.

6- transformers.ipynb : BERT pretrained model and umap reducer for text embedding representation.

7- PassiveAggressive_MLP_Online_fair.ipynb : training PA and MLP reducer, and api.

For more details in slides.




