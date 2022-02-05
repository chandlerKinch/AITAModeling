import pandas as pd
import numpy as np
import string
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from textwrap import wrap
import seaborn as sns
import textstat



class CommentPuller:
    """This class takes a data frame of reddit posts and their associated features.
       This class contains methods for extracting the comments from each post, cleaning post text data and title text.
       It also vectorizers these cleaned data, and puts them into new data frames."""


    #Constructor for the class. If no DataFrame is passed, class is loaded from files.
    def __init__(self, post_df=None, load=False):
        #Check if a dataframe was passed, if not load it from files. 
        if load == False:
            self.post_df = post_df

            #This method will drop rows that contain identical post id's. 
            self._drop_dups()

            self._add_features()
        else:
            self._load()
        

        

    #This method will extract the comments from post_df and make a new dataframe from these comments. 
    def make_comment_df(self):
        try: 
            comments = self.post_df['comments']
        except:
            raise LookupError('Internal post_df has no comments column.')
        try: 
            post_ids = self.post_df['post_id']
        except: 
            raise LookupError('Internal post_df has no post_id column.')

        comment_list = []
        for top_level, post_id in zip(comments, post_ids):
            for comment in top_level.list()[:10]:#Only the top ten comments
                comment_list.append((comment.body, post_id))#Append as a tuple to retain post_id
        self.comment_df = pd.DataFrame(comment_list, columns=['comment', 'post_id']) 

    def label_dfs(self):
        #Call method to label comment_df.
        self._label_comment_df()
        #Call method to label post_df
        self._label_post_df()

    #This method interates through each comment in comment_df and labels them.
    def _label_comment_df(self):
        try:
            self.comment_df
        except:
            raise LookupError('comment_df not found. Call make_comment_df.')
        
        comments = self.comment_df['comment']
        labels = []
        for text in comments:
            table = str.maketrans(dict.fromkeys(string.punctuation))#Remove punc., split, lowercase then check
            text = text.translate(table).lower().split()
            if 'yta' in text and 'nta' in text:#If the string contains both, it will we labeled as NAN
                labels.append(np.NAN)
            elif 'yta' in text:
                labels.append(1)
            elif 'nta' in text:
                labels.append(0)
            else:
                labels.append(np.NAN)#If neither is contained in the string, it will again be labeled as NAN.
        self.comment_df['label'] = labels

    #This method labels each column based on its flair text. Labeled 1 if A-hole, 0 if not. 
    def _label_post_df(self):
        self.post_df['label'] = np.where(self.post_df['flair_text'] == 'Not the A-hole', 0, 1)

    #This method drops rows with that contain indentical post_id's
    def _drop_dups(self):
        try:
            no_dups = self.post_df['post_id'].drop_duplicates()
        except:
            raise LookupError('post_df has no post_id column')
        
        self.post_df = self.post_df.take(no_dups.index)
        self.post_df.reset_index(drop=True ,inplace=True)

    #This method will clean the post_text and title columns of post_df.
    #After this cleaning, these data are vectorized and added to new dataframes.
    def vectorize_text(self, freq=4, gram=(1,1)):
        try:
            self.post_df['post_text']
        except:
            raise LookupError('post_df has no post_text column')

        try:
            self.post_df['title']
        except:
            raise LookupError('post_df has no title column')
        #Clean text to remove numbers, new lines, punc., make lowercase and lemmatize
        post_text = self.post_df.post_text.apply(self._clean_string)

        #Vectorizer the text of the posts, set min_df to freq. 
        text_vectorizer = CountVectorizer(min_df=freq, ngram_range=gram)
        text_vectorizer.fit(post_text)
        text_vector = text_vectorizer.transform(post_text)
        self.text_df = pd.DataFrame(text_vector.toarray(), columns=text_vectorizer.get_feature_names_out())

        #Do the same things as above, put for the title text
        title_text = self.post_df.title.apply(self._clean_string)
        title_vectorizer = CountVectorizer(min_df=freq, ngram_range=gram)
        title_vectorizer.fit(title_text)
        title_vector = title_vectorizer.transform(title_text)
        self.title_df = pd.DataFrame(title_vector.toarray(), columns=title_vectorizer.get_feature_names_out())

        #Check to see if text_df has post_id. 
        #If it doesn't, labels and features will be added. 
        try:
            self.text_df['post_id']
        except:
            #Add id's and features for each row in text_df
            self.text_df.insert(loc=0, column='post_id', value=self.post_df['post_id'])
            self.text_df.insert(loc=1, column='label_id', value=self.post_df['label'])
            self.text_df.insert(loc=2, column='num_words', value=self.post_df['post_text_length'])
            self.text_df.insert(loc=3, column='ease_score', value=self.post_df['text_ease_score'])
            self.text_df.insert(loc=4, column='grade_level', value=self.post_df['text_grade_level'])
            #Add id's and features for each row in title_df
            self.title_df.insert(loc=0, column='post_id', value=self.post_df['post_id'])
            self.title_df.insert(loc=1, column='label_id', value=self.post_df['label'])
            self.title_df.insert(loc=2, column='num_words', value=self.post_df['title_length'])
            self.title_df.insert(loc=3, column='ease_score', value=self.post_df['title_ease_score'])
            self.title_df.insert(loc=4, column='grade_level', value=self.post_df['title_grade_level'])

    #This class removes punctuation, new lines and numbers from a string. Makes string lowercase. 
    #The string is also lemmatized. 
    def _clean_string(self, post):
        #Remove numbers
        text = re.sub(r'\d+', '', post)

        #Remove punc., remove new lines and make lowercase then check
        table = str.maketrans(dict.fromkeys(string.punctuation))
        text = text.translate(table).lower().replace('\n', '')
        #Tokenize text for lemmatizer
        tokens = word_tokenize(text)
        
        #Remove stop words
        no_stop = [word for word in tokens if word not in stopwords.words('english')]
    
        #Intialize lemmatize
        lemmatizer = WordNetLemmatizer()
        #Lemmatize each word in no_stop list and then reassemble into single string for CountVectorizer
        result = " ".join([lemmatizer.lemmatize(w) for w in no_stop])
        return result
    
    #Save class
    def save(self):
        self.post_df.to_pickle('post_df.pkl')
        self.comment_df.to_pickle('comment_df.pkl')
        self.text_df.to_pickle('text_df.pkl')
        self.title_df.to_pickle('title_df')
 
    #Load class
    def _load(self):
        self.post_df = pd.read_pickle('post_df.pkl')
        self.comment_df = pd.read_pickle('comment_df.pkl')
        self.text_df = pd.read_pickle('text_df.pkl')
        self.title_df = pd.read_pickle('title_df')

    #Generate a word cloud for desired dataframe
    def generate_wordcloud(self, name, assholes):
        #Get dataframe as single column of sums and make title
        #Check for whether this cloud will be about assholes or not
        if assholes == False:
            label = 0
        else:
            label=1

        #Check for which dataframe will be wordclouded
        if name == 'title':
            df = self.title_df.iloc[:, 5:].T.sum(axis=1)
            title = "Post title wordcloud"
        elif name == 'text':
            df = self.text_df.iloc[:, 5:].T.sum(axis=1)
            title = "Post body wordcloud"
        else:
            raise ValueError("name must be either title or text")

        #Append title based on if assholes or not
        if assholes == False:
            title += ' of non-assholes'
        else:
            title += ' of assholes'
        
        #Put pieces together to generate word cloud
        wc = WordCloud(width=400, height=330, max_words=150, colormap="Dark2").generate_from_frequencies(df)
        plt.figure(figsize=(10,8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title('\n'.join(wrap(title,60)),fontsize=13)
        plt.show()
    
    #This method will plot the top 25 words from title_df and text_df on a barplot.
    def plot_top25(self, name):
        #Get dataframe as single column of sums and make title
        #Check for which dataframe will be plotted
        if name == 'title':
            df = self.title_df.iloc[:, 5:].T.sum(axis=1)
            title = "Post title top 25 words"
        elif name == 'text':
            df = self.text_df.iloc[:, 5:].T.sum(axis=1)
            title = "Post body top 25 words"
        else:
            raise ValueError("name must be either title or text")

        #Covert back to dataframe so that values can again be columns, then plot.
        temp_df = pd.DataFrame(df.sort_values(ascending=False))
        sns.barplot(data=temp_df.head(25).T)
        plt.title(title)
        plt.xticks(rotation=45)
  
    def _add_features(self):
        #Add features for title
        #Add length of each title
        self.post_df['title_length'] = self.post_df['title'].str.split().str.len()
        #Add ease score according to flesch_reading scale for each title
        self.post_df['title_ease_score'] = self.post_df.apply(lambda row: textstat.flesch_reading_ease(row['title']), axis=1)
        #Add reading level for each title
        self.post_df['title_grade_level'] = self.post_df.apply(lambda row: textstat.gunning_fog(row['title']), axis=1)

        #Add the same stats but for post body now
        #Add length of each body
        self.post_df['post_text_length'] = self.post_df['post_text'].str.split().str.len()
        #Add reading ease score for each text body
        self.post_df['text_ease_score'] = self.post_df.apply(lambda row: textstat.flesch_reading_ease(row['post_text']), axis=1)
        #Adde reading grade level for each text body
        self.post_df['text_grade_level'] = self.post_df.apply(lambda row: textstat.gunning_fog(row['post_text']), axis=1)