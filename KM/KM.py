# importing the modules

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances


# loading the data
df = pd.read_csv('news.csv')


# defining Persian stopwords
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
 'شدند', 'شده', 'شما', 'شند', 'شو', 'شود', 'شون', 'شوند', 'شید', 'طبق', 'طرف', 'طور', 'طول', 'طی', 'ی',
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
 'گونه', 'گیر', 'گیرد', 'گیری', 'یا', 'یابد', 'زاده', 'یافت', 'یافته', 'ها', 'یعنی', 'ای', 'یک', 'یکی']




data = df['news']


tf_idf_vectorizor = TfidfVectorizer(stop_words = persian_stopwords, max_features = 5000)
tf_idf = tf_idf_vectorizor.fit_transform(data)
tf_idf_norm = normalize(tf_idf)
tf_idf_array = tf_idf_norm.toarray()


# defining k-means class

class Kmeans:
    """ 
    K Means Clustering
    
    """
    
    def __init__(self, k, seed = None, max_iter = 200):
        self.k = k
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter
        
            
    
    def initialise_centroids(self, data):
        """
        Randomly Initialise Centroids
    
        """
        
        initial_centroids = np.random.permutation(data.shape[0])[:self.k]
        self.centroids = data[initial_centroids]

        return self.centroids
    
    
    def assign_clusters(self, data):
        """Computing distance of data from clusters 
        and assigning data point to closest cluster     
        """
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        dist_to_centroid =  pairwise_distances(data, self.centroids, metric = 'euclidean')
        self.cluster_labels = np.argmin(dist_to_centroid, axis = 1)
        
        return  self.cluster_labels
    
    
    def update_centroids(self, data):
        """
        Computing average of all data points in cluster
           and assigning new centroids as average of data points
        
        """
        
        self.centroids = np.array([data[self.cluster_labels == i].mean(axis = 0) for i in range(self.k)])
        
        return self.centroids
    
    
    
    def predict(self, data):
        """
        Predict data point cluster
        
        """
        
        return self.assign_clusters(data)
    
    def fit_kmeans(self, data):
        """
        the main loop to fit the algorithm
        Implements initialise centroids and update_centroids
        Returns instance of kmeans class      
            
        """
        self.centroids = self.initialise_centroids(data)
        
        # Main kmeans loop
        for iter in range(self.max_iter):

            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)          

        print("Model finished running")
        return self
    
    
 










# training the model

model = Kmeans(4, 1, 600)
Xf = tf_idf_array
model = model.fit_kmeans(Xf)



# predicting the data
prediction = model.predict(Xf)


# printing the result  
print(np.unique(prediction))
print(prediction[0:1000])