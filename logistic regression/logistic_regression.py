# classification of tweet sentiments

# Importing the libraries

from __future__ import unicode_literals
import numpy as np
import re
from sklearn.datasets import load_files
from hazm import Normalizer, Stemmer


# Loading the dataset

tweets = load_files('tweets/', encoding='utf-8')
X, y = tweets.data, tweets.target


# preprocessing the dataset and building the corpus persian_stopwords_3 0.77

corpus = []

for i in range(len(X)):
    tweet = X[i]
    
# stemming   
    sstl = []
    normalizer = Normalizer()
    stemmer = Stemmer()

    tweet = normalizer.normalize(tweet)
    sent_token_list = tweet.split(' ')
    for tk in sent_token_list:
        sstl.append(stemmer.stem(tk))
    tweet = ' '.join(sstl)
    
    tweet = re.sub(r'https:.*$', ' ', tweet)
    tweet = re.sub(r'[^a-zA-Z0-9ا-یآإأكئؤيء۰-۹_#]',' ', tweet)   

    tweet = re.sub(r'یاشارسلطان', 'یاشار سلطان', tweet)
    tweet = re.sub(r'محمدباقر', 'محمد باقر', tweet)  

    tweet = re.sub(r'\bو', '', tweet)
    tweet = re.sub(r'\b\w{1}\b', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    
    corpus.append(tweet)


persian_stopwords = ['آمد', 'آمده', 'آن', 'آنان', 'آنجا', 'آنها', 'آنچه', 'آنکه',
 'آورد', 'آیا', 'اثر', 'از', 'اس', 'است', 'اش', 'اغلب', 'افراد', 'البت', 'البته', 'اما', 'امر',
 'اند', 'او', 'اکنون', 'ایشان', 'ایشون', 'این', 'اینجا', 'اینک', 'اینکه', 'با', 'بار', 'باز', 'باشد',
 'باشند', 'باعث', 'بالا', 'باید', 'بدون', 'بر', 'برا', 'برای', 'برخ', 'برخی', 'بسیار', 'بسیاری', 'بعد',
 'بعض', 'بعضی', 'بلک', 'بلکه', 'بنابراین', 'به', 'بود', 'بودن', 'بودند', 'بوده', 'بی', 'بیرون', 'بیش',
 'بیشتر', 'بیشتری', 'بین', 'تا', 'تان', 'تاکنون', 'تبدیل', 'تح', 'تحت', 'ترتیب', 'تعداد', 'تعیین',
 'تغییر', 'تما', 'تمام', 'تمامی', 'تنها', 'تهیه', 'تو', 'تون', 'جا', 'جای', 'جایی', 'جدی', 'جز', 'جمع',
 'حال', 'حالا', 'حالی', 'حتی', 'حد', 'حداقل', 'حدود', 'حل', 'خاص', 'خود', 'خودش', 'خویش', 'خیل', 'خیلی',
 'داد', 'دادن', 'دادند', 'داده', 'دارا', 'دارای', 'دارد', 'دارند', 'داری', 'داریم', 'داشت', 'داشتن',
 'داشتند', 'داشته', 'دانس', 'دانست', 'در', 'درباره', 'ده', 'دهد', 'دهند', 'دهه', 'دو', 'دور',
 'دوم', 'دچار', 'دیگر', 'دیگران', 'دیگری', 'را', 'رسید', 'رسیدن', 'رف', 'رفت', 'رو', 'روب', 'روبه', 'روش',
 'روند', 'روی', 'ریز', 'ریزی', 'زیاد', 'زیادی', 'زیر', 'زیرا', 'ساز', 'سازی', 'سال', 'ساله', 'سایر',
 'سبب', 'سراسر', 'سمت', 'سه', 'سهم', 'سو', 'سوم', 'سوی', 'سپس', 'شامل', 'شان', 'شاید', 'شد', 'شدن',
 'شدند', 'شده', 'شما', 'شند', 'شو', 'شود', 'شون', 'شوند', 'شید', 'طبق', 'طرف', 'طور', 'طول', 'طی', 'ع',
 'عدم', 'عل', 'علاوه', 'علت', 'علی', 'علیه', 'عهد', 'عهده', 'عین', 'غیر', 'فرد', 'فردی', 'فقط', 'فوق',
 'فکر', 'قابل', 'قبل', 'لاز', 'لازم', 'لحاظ', 'لذا', 'ما', 'مان', 'مانند', 'متر', 'مثل', 'مد', 'مدت',
 'مربوط', 'مشخص', 'ممکن', 'من', 'مه', 'مهم', 'موجب', 'مورد', 'مون', 'می', 'میآید', 'میان', 'میباشد',
 'میتواند', 'میتوانند', 'میدهد', 'میدهند', 'میرسد', 'میرود', 'میش', 'میشد', 'میشم', 'میشن', 'میشند',
 'میشه', 'میشود', 'میشوم', 'میشوند', 'میشوی', 'میشوید', 'میشویم', 'میشید', 'میکرد', 'میکردند', 'میکن',
 'میکند', 'میکنم', 'میکنه', 'میکنی', 'میکنید', 'میکنیم', 'میگویند', 'میگیرد', 'مییابد', 'نباید', 'نبود',
 'نحو', 'نحوه', 'ندارد', 'ندارند', 'نسب', 'نسبت', 'نشس', 'نشست', 'نظیر', 'نمیشود', 'نه', 'نوع', 'نوعی',
 'نیز', 'نیس', 'نیست', 'نیستند', 'های', 'هایی', 'هر', 'هستند', 'هستی', 'هستیم', 'هم', 'همان', 'همه',
 'همواره', 'همچنان', 'همچنین', 'همچون', 'همیشه', 'همین', 'هنوز', 'هنگا', 'هنگام', 'و', 'وقت', 'وقتی',
 'ول', 'ولی', 'وگو', 'وی', 'پر', 'پس', 'پی', 'پیدا', 'پیش', 'چرا', 'چند', 'چنین', 'چه', 'چون', 'چگونه',
 'چیز', 'چیزی', 'کاملا', 'کدا', 'کدام', 'کرد', 'کردم', 'کردن', 'کردند', 'کرده', 'کس', 'کسان', 'کسانی',
 'کسی', 'کل', 'کلی', 'کم', 'کمی', 'کن', 'کنار', 'کند', 'کنم', 'کنند', 'کننده', 'کنندگان', 'کنی', 'کنید',
 'کنیم', 'که', 'گا', 'گاه', 'گذار', 'گذاری', 'گردد', 'گرف', 'گرفت', 'گرفته', 'گف', 'گفت', 'گفته', 'گون',
 'گونه', 'گیر', 'گیرد', 'گیری', 'یا', 'یابد', 'یاف', 'یافت', 'یافته', 'یعن', 'یعنی', 'ینی', 'یک', 'یکی']



# Training the model

# Making the BOW model

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df = 1, max_df = 0.95, stop_words = persian_stopwords)
X = vectorizer.fit_transform(corpus).toarray()



# making the Tf-Idf Model

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)



# Training the classifier LogisticRegression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(multi_class = 'ovr')
classifier.fit(X_train, y_train)



# training the model on all data

# Training the classifier LogisticRegression

'''from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(multi_class = 'ovr')
classifier.fit(X, y)'''




# Testing model performance

y_pred = classifier.predict(X_test)



# Evaluating the model

# Cross-valitation with 5 folds

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(classifier, X, y, cv = 5)

print('cross-validation scores (5-fold):', cv_scores)
print('mean cross-validation score (5-fold):', round(np.mean(cv_scores), 3))


## confusion matrix, accuracy score, classification report

# Importing evaluation modules

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)



print('\n')
print('1) accuracy:\n\n', round(ac, 2))
print('\n\n')
print('2) confusion matrix:\n\n', cm)
print('\n\n')
print('3) classification report:\n\n', cr)
print('\n')



