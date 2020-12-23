#!/usr/bin/env python
# coding: utf-8

# <h1>Love in the time of Screens </h1>
# <p>December 2020 </p>

# In[38]:


import pandas as pd
#import spacy
from sklearn.feature_extraction.text import CountVectorizer
#nlp = spacy.load("en_core_web_sm")
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


data=pd.read_csv("dataset/data.csv")
data.head()


# In[3]:


#sp = spacy.load('en_core_web_sm')

#stopwords = sp.Defaults.stop_words


# In[4]:


langs=[]
for i in range(data.shape[0]):
    mylang=[]
    x=data.iloc[i]['language'].split(',')
    for j in x:
        if(j not in langs):
            mylang.append(j.strip().split(' '))
    langs.append(mylang)
#langs
data['proc_langs']=langs
data=data.drop(['language'],axis=1)


# In[5]:




# In[6]:


def language_similar(q1,q2,i1,i2):
    if(i1=="not interested"):
        return 0
    else:
        if(len(q2)>1):
            return 1
        else:
            return 0


# In[44]:


def bio_similar(q1,q2):
    try:
        documents=[q1,q2]
        sparse_matrix = count_vectorizer.fit_transform(documents)
        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(doc_term_matrix, 
                      columns=count_vectorizer.get_feature_names(), 
                      index=['q1', 'q1'])
        return cosine_similarity(df, df)[0][1]
    except:
        return 0


# In[48]:


#bio_similar(data['bio'][0],data['bio'][1]) #testing purpose


# In[8]:


def drinks_similar(q1,q2):
    x=0
    y=0
    if(q1=="not at all"):
        x=1
    if(q1=="rarely"):
        x=0.75
    if(q1=="socially"):
        x=0.5
    if(q1=="often"):
        x=0.25
    if(q1=="very often"):
        x=0
    if(q2=="not at all"):
        y=1
    if(q2=="rarely"):
        y=0.75
    if(q2=="socially"):
        y=0.5
    if(q2=="often"):
        y=0.25
    if(q2=="very often"):
        y=0
    return 1-abs(x-y)


# In[9]:


def drugs_similar(q1,q2):
    x=0
    y=0
    if(q1=="sometimes"):
        x=1
    if(q1=="never"):
        x=0
    if(q2=="sometimes"):
        y=1
    if(q2=="never"):
        y=0
    return(1-abs(x-y))


# In[10]:


def job_similar(q1,q2):
    if(q1=='other' or q2=='other'):
        return 0.25
    elif(q1==q2):
        return 1
    else:
        return 0


# In[11]:


def location_similar(q1,q2,c1,c2):
    q1=q1.split(",")
    q2=q2.split(",")
    if(c1=="same city" and q1[0]==q2[0]):
        return 1
    elif(c1=="same state" and q1[1]==q2[1]):
        return 1
    elif(c1=="anywhere"):
        return 1
    else:
        return 0


# In[12]:


def pet_similar(q1,q2):
    dogs1=0
    dogs2=0
    cats1=0
    cats2=0
    if("has dogs"in q1):
        dogs1=1
    if("has dogs"in q2):
        dogs2=1
    if("likes dogs" in q1):
        dogs1=0.5
    if("likes dogs" in q2):
        dogs2=0.5
    if("dislikes dogs" in q1):
        dogs1=-1
    if("dislikes dogs" in q2):
        dogs2=-1
#___________________________________
    if("has cats"in q1):
        cats1=1
    if("has cats"in q2):
        cats2=1
    if("likes cats" in q1):
        cats1=0.5
    if("likes cats" in q2):
        cats2=0.5
    if("dislikes cats" in q1):
        cats1=-1
    if("dislikes cats" in q2):
        cats2=-1
    val1=0
    if(cats1==0 or cats2==0):
        val1=0
    else:
        val1=(1-abs(cats1-cats2))
    if(dogs1==0 or dogs2==0):
        val2=0
    else:
        val2=(1-abs(dogs1-dogs2))
    if(val1!=0 and val2!=0):
        return (val1+val2)/4
    else:
        return val1+val2


# In[13]:


from scipy.spatial import distance
def smokes_similar(q1,q2):
    yes1=0
    yes2=0
    sometimes1=0
    sometimes2=0
    drinking1=0
    drinking2=0
    if("yes" in q1):
        yes1=1
    if("yes" in q2):
        yes2=1
    if("no" in q1):
        yes1=0
    if("no" in q2):
        yes2=0
    if("sometimes" in q1):
        sometimes1=1
        yes1=0.5
    if("sometimes" in q2):
        sometimes2=1
        yes2=0.5
    if("when drinking" in q1):
        drinking1 =0.5
        yes1=0.5
    if("when drinking" in q2):
        drinking2= 0.5
        yes2=0.5
    arr1=[yes1,sometimes1,drinking1]
    arr2=[yes2,sometimes2,drinking2]
    return abs(1-distance.euclidean(arr1,arr2))


# In[14]:


from scipy.spatial import distance
def age_similar(q1,q2):
    diff= distance.euclidean(q1,q2)
    res=(1-diff/q1)
    if(res<0):
        return 0
    else:
        return res


# In[15]:


def status_similar(q1,q2):
    if(q2=="married"):
        return 0
    elif(q2=="seeing someone"):
        return 0.25
    elif(q2=="single"):
        return 0.8
    else:
        return 1


# In[16]:


def orient_similar(s1,s2,o1,o2):
    if(o1=="straight"and o2=="straight"):
        if(s1==s2):
            return 0
        else:
            return 1
    if(o1=="gay" and (o2=="gay" or o2=="bisexual")):
        if(s1==s2):
            return 1
        else:
            return 0
    if(o1=="bisexual" and o2=="bisexual"):
        return 1
    elif(o1=="bisexual" and o2=="gay"):
        if(s1==s2):
            return 1
        else:
            return 0
    else:
        return 0


# In[17]:


def body_similar(q1,q2):
    if(q2 in ['athletic','jacked','curvy','full figured'] or q1==q2):
        return 1
    if(q2 in "fit"):
        return 0.75
    if(q2=="average"):
        return 0.5
    if(q2 in ["a little extra","skinny","rather not say"]):
        return 0.25
    if(q2 in ["used up","overweight","thin"]):
        return 0


# In[18]:


def dropped_similar(q1,q2):
    if(q1==q2):
        return 1
    else:
        return 0


# In[19]:


def education_similar(q1,q2):
    diff= distance.euclidean(q1,q2)
    res=(1-diff/q1)
    return res


# In[20]:


def interest_similar(i1,i2,j1,j2):
    setA=set([i1,i2])
    setB=set([j1,j2])
    res=setA.intersection(setB)
    return len(res)/4


# In[69]:


mydataframe=pd.DataFrame()


# In[78]:


op=pd.DataFrame()
j=0
nmin=9999
nmax=0
for i in reversed(range(data.shape[0])):
    print(int(j)-int(i),j)
    dic={'user_id':data.iloc[i]['user_id']}
    for j in range(data.shape[0]):
        if(data.iloc[i]['user_id']==data.iloc[j]['user_id']):
            dic[data.iloc[j]['user_id']]=0
        else:
            o3=orient_similar(data.iloc[i]['sex'],data.iloc[j]['sex'],data.iloc[i]['orientation'],data.iloc[j]['orientation'])
            if(o3==0):
                dic[data.iloc[j]['user_id']]=0
            else:
                o1=age_similar(data.iloc[i]['age'],data.iloc[j]['age'])
                o2=status_similar(data.iloc[i]['status'],data.iloc[j]['status'])
                o4=drinks_similar(data.iloc[i]['drinks'],data.iloc[j]['drinks'])
                o5=drugs_similar(data.iloc[i]['drugs'],data.iloc[j]['drugs'])
                o6=smokes_similar(data.iloc[i]['smokes'],data.iloc[j]['smokes'])
                o6=job_similar(data.iloc[i]['job'],data.iloc[j]['job'])
                o7=location_similar(data.iloc[i]['location'],data.iloc[j]['location'],data.iloc[i]['location_preference'],data.iloc[j]['location_preference'])
                o8=body_similar(data.iloc[i]['body_profile'],data.iloc[j]['body_profile'])
                o9=dropped_similar(data.iloc[i]['dropped_out'],data.iloc[j]['dropped_out'])
                o10=education_similar(data.iloc[i]['education_level'],data.iloc[j]['education_level'])
                o11=bio_similar(data.iloc[i]['bio'],data.iloc[i]['bio'])  ### Enable only if you have a very high computation power, this process may take many days
                o12=pet_similar(data.iloc[i]['pets'],data.iloc[i]['pets'])
                o13=interest_similar(data.iloc[i]['interests'],data.iloc[i]['other_interests'],data.iloc[j]['interests'],data.iloc[j]['other_interests'])
                o14=language_similar(data.iloc[i]['proc_langs'],data.iloc[j]['proc_langs'],data.iloc[i]['new_languages'],data.iloc[j]['new_languages'])
                res=sum([o1,o2,o4,o5,o6,o8,o9,o10,o11,o12,o13,o14])*o3*o7*(100/12)
                dic[data.iloc[j]['user_id']]=res
                if(nmax<res):
                    namx=res
    op=pd.concat([pd.DataFrame.from_records([dic],index='user_id'),op])


# In[62]:


op.head()


# In[53]:


def Normalize(op): ## Normalising the values in range 0-100
    print("Normalizing Data")
    for i in range((op.shape[0])):
        for j in range((op.shape[0])):
            val=(op.iloc[i][j]-nmin)/(nmax-nmin)
            val_ok=pd.DataFrame([val]).to_numpy()*100
            if(val_ok)<0:
                val_ok=0
            op.iloc[i,j]=val_ok
    return op


# In[65]:


#op=Normalize(op)   ##Normalising the values in range 0-100
#op


# In[58]:


op


# In[ ]:


op.to_csv("Output_file_1.csv")


# In[ ]:


#data[data.user_id=='fffe3500']


# In[ ]:


#data[data.user_id=='fffe3100']


# In[ ]:





# In[ ]:


#data.iloc[0]['proc_langs']


# In[ ]:


#data

