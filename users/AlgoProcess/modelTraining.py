
class Algorithms:



    def RNN(self):
        import numpy as np # linear algebra
        import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
        import seaborn as sns
        import matplotlib.pyplot as plt


        from scipy import stats
        from keras.datasets import imdb
        from keras.preprocessing.sequence import pad_sequences
        from keras.models import Sequential
        from keras.layers import Embedding
        from keras.layers import SimpleRNN,Dense,Activation
        (X_train,Y_train),(X_test,Y_test) = imdb.load_data(path="imdb.npz",num_words=None,skip_top=0,maxlen=None,start_char=1,seed=13,oov_char=2,index_from=3)
        word_index = imdb.get_word_index()
        num_words = 15000
        (X_train,Y_train),(X_test,Y_test) = imdb.load_data(num_words=num_words)
        maxlen=130
        X_train = pad_sequences(X_train, maxlen=maxlen)
        X_test = pad_sequences(X_test, maxlen=maxlen)
        rnn = Sequential()

        rnn.add(Embedding(num_words,32,input_length =len(X_train[0]))) # num_words=15000
        rnn.add(SimpleRNN(16,input_shape = (num_words,maxlen), return_sequences=False,activation="relu"))
        rnn.add(Dense(1)) #flatten
        rnn.add(Activation("sigmoid")) #using sigmoid for binary classification

        print(rnn.summary())
        rnn.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
        history = rnn.fit(X_train,Y_train,validation_data = (X_test,Y_test),epochs = 5,batch_size=128,verbose = 1)
        score,acc = rnn.evaluate(X_test,Y_test)
        score = score
        acc = acc
        return score,acc
        


    def LSTM(self):
        # from sklearn.ensemble import RandomForestRegressor
        import numpy as np # linear algebra
        import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

        from sklearn.feature_extraction.text import CountVectorizer
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        from keras.models import Sequential
        from django.conf import settings
        from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
        from sklearn.model_selection import train_test_split
        from keras.utils import to_categorical
        import re

        path = settings.MEDIA_ROOT + '\\' + 'Sentiment.csv'
        data = pd.read_csv(path)
        # Keeping only the neccessary columns
        data = data[['text','sentiment']]
        data = data[data.sentiment != "Neutral"]
        data['text'] = data['text'].apply(lambda x: x.lower())
        data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

        print(data[ data['sentiment'] == 'Positive'].size)
        print(data[ data['sentiment'] == 'Negative'].size)

        for idx,row in data.iterrows():
            row[0] = row[0].replace('rt',' ')
            
        max_fatures = 2000
        tokenizer = Tokenizer(num_words=max_fatures, split=' ')
        tokenizer.fit_on_texts(data['text'].values)
        X = tokenizer.texts_to_sequences(data['text'].values)
        X = pad_sequences(X)

        # print("b:",self.y_train)
        embed_dim = 128
        lstm_out = 196

        model = Sequential()
        model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        print(model.summary())
        Y = pd.get_dummies(data['sentiment']).values
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
        print(X_train.shape,Y_train.shape)
        print(X_test.shape,Y_test.shape)
        batch_size = 32
        model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)
        validation_size = 1500

        X_validate = X_test[-validation_size:]
        Y_validate = Y_test[-validation_size:]
        X_test = X_test[:-validation_size]
        Y_test = Y_test[:-validation_size]
        score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
        score = score
        acc = acc


        return score,acc  



    def LogisticRegression(self):
        # from sklearn.ensemble import RandomForestRegressor
        import pandas as pd
        from django.conf import settings
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn. metrics import  mean_absolute_error, mean_squared_error
        from sklearn. metrics import  mean_absolute_error, mean_squared_error
        from sklearn. metrics import  mean_absolute_error, mean_squared_error
        from django.conf import settings
        import pandas as pd
        import numpy as np
        path = settings.MEDIA_ROOT + '\\' + 'Twitter_Data.csv'
        data = pd.read_csv(path)



        import re
        TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

        from wordcloud import STOPWORDS
        STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's'])

        def lower(text):
            return text.lower()

        def remove_twitter(text):
            return re.sub(TEXT_CLEANING_RE, ' ', text)

        def remove_stopwords(text):
            return " ".join([word for word in str(text).split() if word not in STOPWORDS])

        def cleantext(text):
            text = lower(text)
            text = remove_twitter(text)
            text = remove_stopwords(text)
            return text

        data = data.dropna()
        data['clean_text'] = data['clean_text'].apply(cleantext)

        from nltk.stem import WordNetLemmatizer

        lematizer=WordNetLemmatizer()

        def lemmatizer_words(text):
            return " ".join([lematizer.lemmatize(word) for word in text.split()])

        # data['clean_text']=data['clean_text'].apply(lambda text: lemmatizer_words(text))  

        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test = train_test_split(data['clean_text'],data['category'],test_size=0.2,random_state=45)
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer()
        x_train_vec = tf.fit_transform(x_train)
        x_test_vec = tf.transform(x_test)
        print('hey its here')
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        model = LogisticRegression()
        model.fit(x_train_vec, y_train)
        y_pred = model.predict(x_test_vec)
        cr_log = classification_report(y_pred, y_test, output_dict=True)
        cr_log = pd.DataFrame(cr_log).transpose()
        cr_log = pd.DataFrame(cr_log)
        cr_log = cr_log.to_html
        return cr_log



    def GaussinNB(self):
        import nltk, warnings, string
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
        from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.neural_network import MLPClassifier
        from django.conf import settings
        from sklearn.feature_extraction.text import TfidfVectorizer
        from nltk.corpus import stopwords
        from nltk.stem import SnowballStemmer, WordNetLemmatizer
        from wordcloud import WordCloud
        import re
        from sklearn.model_selection import train_test_split
        path = settings.MEDIA_ROOT + '\\' + 'Twitter_Data.csv'
        data = pd.read_csv(path)
        data.dropna(inplace=True)



        import re
        TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

        from wordcloud import STOPWORDS
        STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's'])

        def lower(text):
            return text.lower()

        def remove_twitter(text):
            return re.sub(TEXT_CLEANING_RE, ' ', text)

        def remove_stopwords(text):
            return " ".join([word for word in str(text).split() if word not in STOPWORDS])

        def cleantext(text):
            text = lower(text)
            text = remove_twitter(text)
            text = remove_stopwords(text)
            return text

        data = data.dropna()
        data['clean_text'] = data['clean_text'].apply(cleantext)

        from nltk.stem import WordNetLemmatizer

        lematizer=WordNetLemmatizer()

        def lemmatizer_words(text):
            return " ".join([lematizer.lemmatize(word) for word in text.split()])

        data['clean_text']=data['clean_text'].apply(lambda text: lemmatizer_words(text))
        X = data.clean_text
        y = data.category  

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)      
        tfidf = TfidfVectorizer()
        X_train_vect = tfidf.fit_transform(X_train)
        X_test_vect = tfidf.transform(X_test)
        from sklearn.naive_bayes import GaussianNB
        # from sklearn.metrics import accuracy_score
        mnb = MultinomialNB()
        mnb.fit(X_train_vect,y_train)
        mnb_pred = mnb.predict(X_test_vect)
        
        from sklearn.metrics import accuracy_score,classification_report
        cr_nv = classification_report(mnb_pred, y_test, output_dict=True)
        cr_nv = pd.DataFrame(cr_nv).transpose()
        cr_nv = pd.DataFrame(cr_nv)
        cr_nv = cr_nv.to_html
        return cr_nv

    def DecisionTree(self):
        import nltk, warnings, string
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
        from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.neural_network import MLPClassifier
        from django.conf import settings
        from sklearn.feature_extraction.text import TfidfVectorizer
        from nltk.corpus import stopwords
        from nltk.stem import SnowballStemmer, WordNetLemmatizer
        from wordcloud import WordCloud
        import re
        from sklearn.model_selection import train_test_split
        path = settings.MEDIA_ROOT + '\\' + 'Twitter_Data.csv'
        data = pd.read_csv(path)
        data.dropna(inplace=True)



        import re
        TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

        from wordcloud import STOPWORDS
        STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's'])

        def lower(text):
            return text.lower()

        def remove_twitter(text):
            return re.sub(TEXT_CLEANING_RE, ' ', text)

        def remove_stopwords(text):
            return " ".join([word for word in str(text).split() if word not in STOPWORDS])

        def cleantext(text):
            text = lower(text)
            text = remove_twitter(text)
            text = remove_stopwords(text)
            return text

        data = data.dropna()
        data['clean_text'] = data['clean_text'].apply(cleantext)

        from nltk.stem import WordNetLemmatizer

        lematizer=WordNetLemmatizer()

        def lemmatizer_words(text):
            return " ".join([lematizer.lemmatize(word) for word in text.split()])

        data['clean_text']=data['clean_text'].apply(lambda text: lemmatizer_words(text))
        X = data.clean_text
        y = data.category  

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)        
        tfidf = TfidfVectorizer()
        X_train_vect = tfidf.fit_transform(X_train)
        X_test_vect = tfidf.transform(X_test)
        
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier()
        dt.fit(X_train_vect,y_train)
        dt_pred = dt.predict(X_test_vect)
        from sklearn.metrics import accuracy_score,classification_report
        cr_dc = classification_report(dt_pred, y_test, output_dict=True)
        cr_dc = pd.DataFrame(cr_dc).transpose()
        cr_dc = pd.DataFrame(cr_dc)
        cr_dc = cr_dc.to_html
        return cr_dc


    def predict(self,posting):
        import nltk, warnings, string
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
        from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
        from sklearn.naive_bayes import MultinomialNB
        from django.conf import settings
        from sklearn.neural_network import MLPClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from nltk.corpus import stopwords
        from nltk.stem import SnowballStemmer, WordNetLemmatizer
        from wordcloud import WordCloud
        import re
        from sklearn.model_selection import train_test_split
        path = settings.MEDIA_ROOT + '\\' + 'Twitter_Data.csv'
        data = pd.read_csv(path)
        data.dropna(inplace=True)



        import re
        TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

        from wordcloud import STOPWORDS
        STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's'])

        def lower(text):
            return text.lower()

        def remove_twitter(text):
            return re.sub(TEXT_CLEANING_RE, ' ', text)

        def remove_stopwords(text):
            return " ".join([word for word in str(text).split() if word not in STOPWORDS])

        def cleantext(text):
            text = lower(text)
            text = remove_twitter(text)
            text = remove_stopwords(text)
            return text

        data = data.dropna()
        data['clean_text'] = data['clean_text'].apply(cleantext)

        from nltk.stem import WordNetLemmatizer

        lematizer=WordNetLemmatizer()

        def lemmatizer_words(text):
            return " ".join([lematizer.lemmatize(word) for word in text.split()])

        data['clean_text']=data['clean_text'].apply(lambda text: lemmatizer_words(text))
        X = data.clean_text
        y = data.category  

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)        
        tfidf = TfidfVectorizer()
        X_train_vect = tfidf.fit_transform(X_train)
        X_test_vect = tfidf.transform(X_test)
        nb = LogisticRegression()
        nb.fit(X_train_vect,y_train)

        # input_text = ["customer service associate us, ca, san francisco novitex enterprise solutions, formerly pitney bowes management services, delivers innovative document communications management solutions help companies around world drive business process efficiencies, increase productivity, reduce costs improve customer satisfaction. almost 30 years, clients turned us integrate optimize enterprise-wide business processes empower employees, increase productivity maximize results. trusted partner, continually focus delivering secure, technology-enabled document communications solutions improve clients' work processes, enhance customer interactions drive growth. customer service associate based san francisco, ca. right candidate integral part talented team, supporting continued growth.responsibilities:perform various mail center activities (sorting, metering, folding, inserting, delivery, pickup, etc.)lift heavy boxes, files paper neededmaintain highest levels customer care demonstrating friendly cooperative attitudedemonstrate flexibility satisfying customer demands high volume, production environmentconsistently adhere business procedure guidelinesadhere safety procedurestake direction supervisor site managermaintain logs reporting documentation; attention detailparticipate cross-training perform duties assigned (filing, outgoing shipments, etc)operating mailing, copy scanning equipmentshipping &amp; receivinghandle time-sensitive material like confidential, urgent packagesperform tasks assignedscanning incoming mail recipientsperform file purges pullscreate files ship filesprovide backfill neededenter information daily spreadsheetsidentify charges match billingsort deliver mail, small packages minimum requirements:minimum 6 months customer service related experience requiredhigh school diploma equivalent (ged) requiredpreferred qualifications:keyboarding windows environment pc skills required (word, excel powerpoint preferred)experience running mail posting equipment plusexcellent communication skills verbal writtenlifting 55 lbs without accommodationswillingness availability work additional hours assignedwillingness submit pre-employment drug screening criminal background checkability effectively work individually team environmentcompetency performing multiple functional tasksability meet employer's attendance policy computer software"]
        input_text = [posting]
        

        input_data_features = tfidf.transform(input_text)
        # making prediction
        prediction = nb.predict(input_data_features)
        print(prediction)
        if (prediction[0] == -1):
            result = 'youy tweet is negative'
        elif (prediction[0] == 0):
            result = 'your tweet is neutral'
        else:
            result = 'your tweet is positive'
        return result