
# Data Science 1: <u> MIDTERM WRITEUP</u> 
***
## *Jarvis*: a slackbot
### David Landay, MS CSDS

##### What is Jarvis?
Jarvis is a **<a href='https://en.wikipedia.org/wiki/Chatbot'>chatbot</a>** $\rightarrow$ a computer program that responds to user input in a way that simulates human-to-human interactions. That interaction is typically in the form of text, auditory speech, or an appropriate response to visual input(ie: images).<br><br>
Jarvis utelizes the SciKitLearn python package and the <a href='https://slack.com/'>Slack</a> API to respond to user input over a Slack message channel. A user can train Jarvis to categorize, or classify, a specific set of text and then predict future user input. In this project, Jarvis functions off of a supervised learning text classifier. In otherwords, in order to hold a conversation with a user, it must train to do so from text that a user provides.<br><br>
In order to train,Jarvis is put into `training mode` when the user specifies. In particular, when a user sends the message *training time* over Slack. Jarvis will then prompt the user to indicate how the training data should be classified. The five target classifications that Jarvis is equipped to handle include:<br><br>
* TIME $\rightarrow$ for any requests/references the user makes to time. *ex: What is the current time?*
* GREET $\rightarrow$ for any user input that references a greeting or introduction. *ex: Hi, how are you?*
* PIZZA $\rightarrow$ for any user input that references buying, eating, consuming etc... pizza. *ex: I'm hungry!*
* WEATHER $\rightarrow$ for any user input regarding the current state of the weather. *ex: what is the weather like?*
* JOKE $\rightarrow$ when the user prompts Jarvis to tell a joke. *ex: tell me a joke please.*

When the user types *done* Jarvis should exit training mode, and is now equipped to predict future prompts from the user. An example of this process is shown below:

<tr>
    <td><img src="ProcessScreenShots/Actual_Training.png" alt="Training" style="width: 350px;height: 450px"/></td>
    <td><img src="ProcessScreenShots/Testing_Actual.png" alt="Testing" style="width: 350px;height: 450px"/></td>
</tr>

Similar to training Jarvis, when testing it's ability to classify text, the user need only type *testing time*. Jarvis will train it's "Brain" - a SkLearn pipeline for handling text input utilizing Naive Bayes as a predictive model - on the data provided in training mode (which has been stored in an SQL database).

***

##### How Well Does Jarvis Work:

Jarvis works quite well in testing. From the above screenshot, Jarvis correctly classified all of the input that it was given. However, as the training database was built from example text believed to cover a general spectrum of questions and phrases relating to each class, I was fairly uncreative when I tested Jarvis initially. For example, in the above screenshot, I trained Jarvis to recognize fairly generic ways of asking for the time. It is apparent that I tested Jarvis' ability using a training example, which is technically an unfair way to test Jarvis' ability.<br><br> The below example shows an example of abstraction; asking for the time in a philosophical way:

<img src="ProcessScreenShots/Abstraction_test.png" alt="Training" style="width: 350px;height: 450px"/>

Notice that Jarvis is very sensitive to the language it is given; it doesn't properly classify "when" until it recognizes word structure that it has seen before in the `TIME` corpus. Since Jarvis' brain classifies based on word frequency, "when" must appear infrequently among all training corpera. This indicates that Jarvis' ability to predict begins to decay when new words are introduced, or when given short documents. The length of the document matters because if the only new word is "when," then the classifier looks at the remaining words. If the documents given in training are short, then there are fewer opportunities for a word that should be common to all documents (words like: the, a, are, etc... - "linguistic glue" that carry no topical content) to be considered as such by Jarvis' "Brain". Hence, Jarvis can be swayed by the length of the document given to it.

<img src="ProcessScreenShots/One_Letter.png" alt="Training" style="width: 350px;height: 350px"/>

I tried typing individual letters to see how Jarvis would react. Based on the above, it could be that the training data is skewed towards `PIZZA`.

***

##### How Can Jarvis Improve?

The most obvious way to improve Jarvis is to give it more training data. The more sets of word combinations (documents) Jarvis trains on, the better Jarvis' performance becomes. More data to train on means that Jarvis will be able to classify more accurately against a greater variety of words you throw at it. In addition, as stated above, training Jarvis with larger documents will increase the frequency of common words, allowing Jarvis' "Brain" to better predict from unique words.<br><br>
Otherwise, Jarvis' "Brain" can always function using a different learning algorithm. For this project, Jarvis uses a <a href='https://en.wikipedia.org/wiki/Bag-of-words_model'>Bag-of-words</a> (BOW) model. Each slack message that the user provides a bag-of-words (a multiset of words) in which the multiplicity of each word in the set is being kept track of. For the training data, each class is built from 8-10 messages/documents in which the frequency of distinct words is being stored. The word frequencies are then converted into weights:
<img src="https://deeplearning4j.org/img/tfidf.png" alt="Training"/>

This is saying that the weight of a word in a specific document is determined by comparing the frequency of a word to the number of documents in the corpus containing the word. If many of the documents in the corpus contain the word, then that word carries less weight in the prediction of a class. <br><br> 
So, word counts are turned into word relevances, which become the features we feed into the classifier. In training mode, the documents given to each corpus are transformed into these weight vectors (vectors containing the relevances of each word in the corpus). So, each document is now an array of weights: if a word (from the possible unique words in the corpus) is not present in the document, it's array element is 0, else its array element is the associated weight given by the above equation. The documents are then fit to a Multinomial Naive Bayes model. This model computes the probability of a class (how many documents are given a class of `TIME`,`PIZZA`,`GREET` ... etc out of all the documents in the training database), then caculates the probability of a word from the corpus being from a particular class: P($word_i$ | $class_i$) = $\frac{n_k +1}{n|vocabulary|}$ Where, $n$ is the number of unique words found in the class, $n_k$ is the number of times $word_k$ appears in documents associated with a class, and |vocabulary| is the size of the corpus. Finally, when you provide Jarvis with a message in testing mode, Jarvis will determine the probability of each word in the message belonging to a particular class. The product of all of those probabilities is computed for each class, and then whichever product is greatest yields the predicted outcome. In the case where a word in the message does not belong to the database, i.e a word that has never been seen by Jarvis, a very small probability is given to that word (so that the product is not zero).<br><br> Hence, it is now understood why the length of training documents matters, and why Jarvis predicted incorrectly for more abstract user input. The database of documents I made is fairly small, meaning the number of unique words could be small. Perhaps there is a better text classification model that will handle sparse data better than Multinomial Naive Bayes. Also, instead of looking at a Bag-of-words, which is essentially an N-gram model looking at 1-grams, an N-gram model could be explored. Using sklearn, this is as simple as changing the N-gram parameter to be a value greater than 1 in the `ngram_range` parameter space.

***

##### Jarvis' Predictive Power!

Built into the Sklearn package are modules to help visualize and determine how a particular model is predicting.
To check how well Jarvis is performing, we can calculate it's accuracy score by measuring the ratio of correct classifications to the total number of classifications. To do this, I will need to build a test set of documents for Jarvis to predict on. I will input these as features to the classifier, but have an accompanying set of target values (the actual classes) associated with each document. A comparison will be made between the number of correctly identified classes and the total number of test samples given to Jarvis.


```python
import pickle
import sklearn
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from random import shuffle
```


```python
#Load in the pickled brain:
brain_in = open('jarvis_brain.pkl','rb')
BRAIN = pickle.load(brain_in)

#test to see if it worked:
print(BRAIN)
```

    Pipeline(steps=[('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip...inear_tf=False, use_idf=True)), ('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))])
    

Beautiful... Here is a test set and the analysis:


```python
#establish connection to db:
conn = sqlite3.connect('jarvis.db')
c = conn.cursor()

#define the data:
Data = []
X = []
y = []

#un-sort the rows (will be needed for cross-validation later)
for row in c.execute("SELECT * from training_data"):
    Data.append(row)
shuffle(Data)
for row in Data:
    X.append(row[1])
    y.append(row[2])
    
#Train the brain on the data:
BRAIN.fit(X,y)

#Define the test sets:
x_test = ['what does your watch read?','tickle me','how you livin?','gimme a pie','I want to laugh',
          'don\'t forget your hat','is it chilly?','Will I be late?','I want to feast','say hello',
          'when will my brother be here?','my stomach is rumbling','how hot is it today?','rolling on the floor laughing']
y_test = ['TIME','JOKE','GREET','PIZZA','JOKE','WEATHER','WEATHER','TIME','PIZZA','GREET','TIME','PIZZA','WEATHER','JOKE']


#Create the predicted y:
y_pred = BRAIN.predict(x_test)

#show accuracy scores:
print(y_pred)
accuracy_score(y_test,y_pred)
```

    ['TIME' 'JOKE' 'GREET' 'PIZZA' 'PIZZA' 'GREET' 'WEATHER' 'WEATHER' 'PIZZA'
     'GREET' 'WEATHER' 'WEATHER' 'WEATHER' 'WEATHER']
    




    0.5714285714285714



Eww... I have clearly built Jarvis with little thought.To understand why Jarvis' predictive power is so weak, it is important to explore the database that was created in training mode:


```python
for row in c.execute("SELECT * from training_data LIMIT 18"):
    print(row)
```

    (1, 'what time is it?', 'TIME')
    (2, 'what time do you have?', 'TIME')
    (3, 'what does the clock say?', 'TIME')
    (4, 'what hour is it?', 'TIME')
    (5, 'what time it be?', 'TIME')
    (6, 'what time ya got?', 'TIME')
    (7, 'time?', 'TIME')
    (8, 'what time?', 'TIME')
    (9, 'order me a pizza', 'PIZZA')
    (10, 'get pizza', 'PIZZA')
    (11, 'pizza', 'PIZZA')
    (12, 'I want pizza', 'PIZZA')
    (13, 'I am hungry', 'PIZZA')
    (14, 'need food', 'PIZZA')
    (15, 'I would really like something to eat', 'PIZZA')
    (16, 'I would like some dinner', 'PIZZA')
    (17, 'I want lunch', 'PIZZA')
    (18, 'breakfast now!', 'PIZZA')
    

The output above is evidence that Jarvis trained on a lousy training set. Lousy in the sense that there is little variety among the words in some of the documents. Here, the `TIME` and `PIZZA` entries were selected because the input documents show a good contrast amongst the words used to describe each class. For example, most of the entries for `TIME` include the words "what", "is", and "it". As discussed previously, these words don't define context as they are common amongst all documents in the corpus.Thus, the unique words under `TIME` are significantly reduced: "time", "do", "be", "ya" (although this is a mispelling of you - a non-topical word). Keep in mind that as the database gets built further, these words could be less and less unique to the class `TIME`. Effectively, the only word left to identify a document as `TIME` is the word "time". Compare this to the documents that train `PIZZA`: "pizza", "hungry", "order", "lunch", "breakfast", "food"...etc. There are many more words that could be used to identify a document as `PIZZA`. In essence, it is undesireable to define a class where the only correlated word is the name of the class itself. 
***

So, the way that I scored Jarvis above is an ok measure of how well Jarvis is modelling in general. But, there are infinitely many documents that can be passed to Jarvis, so the accuracy score for the above test set  is not a good representation of Jarvis' performance in general.<br><br>
A better way to test Jarvis is to take the database and randomly split that into a training and testing set of documents, and then calculate the accuracy scores. In this way we are better assessing how well the bag-of-words model was constructed.


```python
#perform k-fold crossvalidation:
scores = cross_val_score(BRAIN,X,y, cv=6)
print(scores)
```

    [ 0.6         0.5         0.66666667  0.83333333  0.8         0.8       ]
    

The cross_val_score module takes a classifier (in this case Jarvis' "Brain") and the labeled data set (X,y) and performs a k-fold cross validation. I opted to perform 6 cross validation tests in which testing data was witheld. In the same way as before, the score is then calculated against the witheld testing data.


```python
#compute the mean score and confidence interval of the score estimate
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    Accuracy: 0.70 (+/- 0.24)
    

This metric is saying that we are 95% confident that the mean accuracy score for prediction under the database I have constructed for Jarvis is between 0.46 and 0.94 - quite the spread (to be honest I'm not sure how sklearn is handling the upperbound because in earlier trials I had unreasonable confidence intervals). <br><br>
In conclusion, Jarvis requires much more training data to be a good conversationalist. Overall, Jarvis is not doing a terrible job of predicting context, but based on the results, better training examples will improve Jarvis' performance.

***

## "Post-Mortem":

This project was not only fun, but offered practical learning experience. Before this, I had very few encounters with class based programming. At the beginning, I struggled to understand the logic of the jarvis.py program; in particular, the conventions of *self*. But, I went through the functions step-by-step and soon realized the simplicity and power of programming in this fashion. In addition, I struggled at first to understand how Jarvis should talk back to the user through Slack messages. Once the logic of programming the conversation became clear, one difficulty I faced was ensuring that once I told Jarvis the name of an Action in training mode, that the conversation logic did not jump to the next step in the training process. I figured out that I could trick the logic by forcing Jarvis to think that I was typing a message. This was crucial in implementing the proper algorithm.<br><br>
Like most students, I also learned how to build an sql database (it's actually really simple and powerful). In addition, I learned the sklearn pipeline and pickling, two tools that made implementing and serializing Jarvis' brain very easy (three lines of code).Finally, I became more familiar with Slack and APIs.<br> For the future, I would like to build up Jarvis' database and possibly try different learning models.
