# amazon-review-utils
Utilities for preprocessing and analyzing [Amazon review data](http://jmcauley.ucsd.edu/data/amazon/).  Includes a Naive Bayes sentiment model.

Dependencies: [nltk](http://www.nltk.org/), [WordNetLemmatizer](http://www.nltk.org/_modules/nltk/stem/wordnet.html), [numpy](http://www.numpy.org/).  To download the corpus used by WordNetLemmatizer, do this in python:

```python
import nltk
nltk.download("wordnet", "/path/to/save/nltk_data")
```


# Usage examples
Make a stemmed and lemmatized bag-of-words dataset:

```python
import amazon_review_utils as azu
D = azu.make_dataset('reviews_Amazon_Instant_Video_5.json.gz')
```

Train a naive Bayes model and predict the sentiment-polarity of comments:
```python
import amazon_review_utils as azu
from sklearn.model_selection import train_test_split
from pprint import pprint as pp
import numpy as np

# load and preprocess data
data_path = './reviews_Amazon_Instant_Video_5.json.gz'
D = azu.make_dataset(data_path)

# define training and testing sets
Xall = D['comments']
yall = D['ratings']
Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.33)

# fit and predict
M = azu.NaiveBayes()
M.fit(Xtr,ytr,verbose=True)
yM = M.predict(Xte)
```

Print classification accuracies:
```python
from sklearn.metrics import accuracy_score, confusion_matrix

print('accuracy = %f' % (accuracy_score(yte,yM)))
C = confusion_matrix(yte, yM, labels=['pos','neg'])        # C_ij = number of times labels[i] was predicted to be labels[j]
p0 = C[0,0]/np.sum(C[0,:])
p1 = C[1,1]/np.sum(C[1,:])
cn_accuracy = (p0+p1)/2
print('class-normalized accuracy = %f' % cn_accuracy)
print('confusion matrix = ')
pp(C)
```
```bash
accuracy = 0.879040
class-normalized accuracy = 0.769687
confusion matrix = 
array([[5012,  526],
       [ 215,  373]])
```

Print word sentiment scores:
```python
sentiment_scores = M.dllikelihoods
for w in sorted(sentiment_scores, key=sentiment_scores.get, reverse=True):
    print(w, sentiment_scores[w])
```
```bash
great 1.14444556944
love 1.1016236715
season 1.05951238149
enjoy 0.930963929036
series 0.911828128291
...(omitted for brevity)...
now 0.000457325680016
a 0.000268640025982
some -0.00413303350801
where -0.00500446217099
i -0.00869491592333
...(omitted for brevity)...
didnt -0.617309841741
movie -0.68408527281
why -0.730638527048
no -0.754135658315
bad -0.865642160456
```
