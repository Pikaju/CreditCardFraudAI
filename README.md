# AI based credit card fraud detection

Using TensorFlow 2.0 and Keras to detect credit card fraud based on the [Kaggle dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
The dataset consists of credit card transactions. Each transaction has a timestamp, an amount (i.e. how much money
was to be transferred), as well as 28 floating point values that are a result of principal component analysis (PCA)
to anonymize the users. The dataset also includes information about which transactions are fraudulent.

The data distribution is skewed. Only 0.172% of the transactions are fraudulent, meaning a model which classifies
every transaction as legitimate has an accuracy of 99.828%. 

The data is first preprocessed: Timestamps are completely disregarded, and the amounts are scaled logarithmically,
as they can get very large and may disrupt the learning process.

A simple multi-layer-perceptron (MLP) neural network can already achieve fairly good results, reaching roughly
99.95% - 99.96% accuracy.
