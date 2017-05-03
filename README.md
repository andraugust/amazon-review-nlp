# amazon_review-utils
Utilities for preprocessing and analyzing [Amazon review data](http://jmcauley.ucsd.edu/data/amazon/).  Includes a Naive Bayes sentiment model.

Dependencies: [nltk](http://www.nltk.org/), [WordNetLemmatizer](http://www.nltk.org/_modules/nltk/stem/wordnet.html), [numpy](http://www.numpy.org/).

To download the WordNet corpus, do this in python:

```python
import nltk
nltk.download("wordnet", "/path/to/save/nltk_data")
```


# Usage example
Make a stemmed and lemmatized bag-of-words dataset:



```bash
$ python naive_bayes_sentiment.py


```
