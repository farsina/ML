# importing the modules

import numpy as np
import pandas as pd
import math
from scipy.stats import multivariate_normal as mvn
from numpy.linalg import slogdet, det, solve
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


# loading the data
df = pd.read_csv('news.csv')


X = df["news"]



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




tf_idf_vectorizor = TfidfVectorizer(stop_words = persian_stopwords, max_features = 20000)
                             
tf_idf = tf_idf_vectorizor.fit_transform(data)
tf_idf_norm = normalize(tf_idf)
tf_idf_array = tf_idf_norm.toarray()



# defining Gaussian mixture models class

class GMM:
    """
    Gaussian Mixture Model
       
    """
    def __init__(self, C, n_runs):
        self.C = C # number of Guassians or clusters
        self.n_runs = n_runs
        
    
    def get_params(self):
        return (self.mu, self.pi, self.sigma)
    
    
        
    def calculate_mean_covariance(self, X, prediction):
        """
        Calculate means and covariance of different
            clusters from k-means prediction
        
        """
        d = X.shape[1]
        labels = np.unique(prediction)
        self.initial_means = np.zeros((self.C, d))
        self.initial_cov = np.zeros((self.C, d, d))
        self.initial_pi = np.zeros(self.C)
        
        counter=0
        for label in labels:
            ids = np.where(prediction == label) # returns indices
            self.initial_pi[counter] = len(ids[0]) / X.shape[0]
            self.initial_means[counter,:] = np.mean(X[ids], axis = 0)
            de_meaned = X[ids] - self.initial_means[counter,:]
            Nk = X[ids].shape[0] # number of data points in current gaussian
            self.initial_cov[counter,:, :] = np.dot(self.initial_pi[counter] * de_meaned.T, de_meaned) / Nk
            counter+=1
        assert np.sum(self.initial_pi) == 1    
            
        return (self.initial_means, self.initial_cov, self.initial_pi)
    
    
    
    def _initialise_parameters(self, X):
        """
        Implement k-means to find starting parameter values

        """
        n_clusters = self.C
        kmeans = KMeans(n_clusters= n_clusters, init="k-means++", max_iter=500, algorithm = 'auto')
        fitted = kmeans.fit(X)
        prediction = kmeans.predict(X)
        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(X, prediction)
        
        
        return (self._initial_means, self._initial_cov, self._initial_pi)
            
        
        
    def _e_step(self, X, pi, mu, sigma):
        """
        Performs E-step on GMM model

        """
        N = X.shape[0] 
        self.gamma = np.zeros((N, self.C))

        const_c = np.zeros(self.C)
        
        
        self.mu = self.mu if self._initial_means is None else self._initial_means
        self.pi = self.pi if self._initial_pi is None else self._initial_pi
        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov

        for c in range(self.C):
            # Posterior Distribution using Bayes Rule
            self.gamma[:,c] = self.pi[c] * mvn.pdf(X, self.mu[c,:], self.sigma[c])

        # normalize across columns to make a valid probability
        gamma_norm = np.sum(self.gamma, axis=1)[:,np.newaxis]
        self.gamma /= gamma_norm

        return self.gamma
    
    
    def _m_step(self, X, gamma):
        """
        Performs M-step of the GMM
        We need to update our priors, our means
        and our covariance matrix.

        """
        N = X.shape[0] # number of objects
        C = self.gamma.shape[1] # number of clusters
        d = X.shape[1] # dimension of each object

        # responsibilities for each gaussian
        self.pi = np.mean(self.gamma, axis = 0)

        self.mu = np.dot(self.gamma.T, X) / np.sum(self.gamma, axis = 0)[:,np.newaxis]

        for c in range(C):
            x = X - self.mu[c, :] # (N x d)
            
            gamma_diag = np.diag(self.gamma[:,c])
            x_mu = np.matrix(x)
            gamma_diag = np.matrix(gamma_diag)

            sigma_c = x.T * gamma_diag * x
            self.sigma[c,:,:]=(sigma_c) / np.sum(self.gamma, axis = 0)[:,np.newaxis][c]

        return self.pi, self.mu, self.sigma
    
    
    def _compute_loss_function(self, X, pi, mu, sigma):
        """
        Computes lower bound loss function
        
        """
        N = X.shape[0]
        C = self.gamma.shape[1]
        self.loss = np.zeros((N, C))

        for c in range(C):
            dist = mvn(self.mu[c], self.sigma[c],allow_singular=True)
            self.loss[:,c] = self.gamma[:,c] * (np.log(self.pi[c]+0.00001)+dist.logpdf(X)-np.log(self.gamma[:,c]+0.000001))
        self.loss = np.sum(self.loss)
        return self.loss
    
    
    
    def fit(self, X):
        """Compute the E-step and M-step and
            Calculates the lowerbound
        
        Parameters:
        -----------
        X: (N x d), data 
        
        Returns:
        ----------
        instance of GMM
        
        """
        
        d = X.shape[1]
        self.mu, self.sigma, self.pi =  self._initialise_parameters(X)
        
        try:
            for run in range(self.n_runs):  
                self.gamma  = self._e_step(X, self.mu, self.pi, self.sigma)
                self.pi, self.mu, self.sigma = self._m_step(X, self.gamma)
                loss = self._compute_loss_function(X, self.pi, self.mu, self.sigma)
                


        
        except Exception as e:
            print(e)
            
        
        return self
    
    
    
    
    def predict(self, X):
        """
        Returns predicted labels using Bayes Rule to
        Calculate the posterior distribution
        
        """
        labels = np.zeros((X.shape[0], self.C))
        
        for c in range(self.C):
            labels [:,c] = self.pi[c] * mvn.pdf(X, self.mu[c,:], self.sigma[c])
        labels  = labels .argmax(1)
        return labels 
    
    def predict_proba(self, X):
        """
        Returns predicted labels
        
        """
        post_proba = np.zeros((X.shape[0], self.C))
        
        for c in range(self.C):
            # Posterior Distribution using Bayes Rule, try and vectorise
            post_proba[:,c] = self.pi[c] * mvn.pdf(X, self.mu[c,:], self.sigma[c])
    
        return post_proba
        
sklearn_pca = PCA(n_components = 2)












# training the model

model = GMM(4, n_runs = 100)
Xf = sklearn_pca.fit_transform(tf_idf_array)
model = model.fit(Xf)





# predicting the data
prediction = model.predict(Xf)



# printing the result  
print(np.unique(prediction))
print(prediction[0:1000])       


        