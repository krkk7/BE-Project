from datetime import date, datetime
from pydoc import doc
import random
import warnings
warnings.filterwarnings('ignore')

import re
import string

import json
from collections.abc import MutableMapping
import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,f1_score,accuracy_score,confusion_matrix
from sklearn.metrics import roc_curve,auc,roc_auc_score
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter

from flask import Flask , render_template, request, redirect, url_for, session, flash
from flask import Flask, jsonify, request
import numpy as np
import  pyrebase

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import csv



df = pd.read_csv('course.csv')
selected_features=['Name','Type','Category','Subcategory','Course_organization']

for feature in selected_features:
    df[feature]=df[feature].fillna('')

combined_features=df['Name']+''+df['Type']+''+df['Category']+''+df['Subcategory']+''+df['Course_organization']
vectorizer=TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)
similarity=cosine_similarity(feature_vectors)



cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

firebaseConfig = {
  'apiKey': "AIzaSyBdTOttW-9qyDt7YQYOENqwvy9BwdUlKDE",
  'authDomain': "studypress-4af82.firebaseapp.com",
  'projectId': "sstudypress-4af82",
  'storageBucket': "studypress-4af82.appspot.com",
  'messagingSenderId': "633006789601",
  'appId': "1:633006789601:web:f2dd08788b263d476038a0",
  'measurementId': "G-FSJQR286FC",
  'databaseURL':""
}

firebase=pyrebase.initialize_app(firebaseConfig)

auth=firebase.auth()
storage=firebase.storage()
db=firestore.client()


app=Flask(__name__)

app.secret_key="kesavan99"






@app.route("/home")
def home():
    user=session["user"]
    id=user["localId"]


    relt=db.collection('users').document(id).get()
    result=relt.to_dict()
    if(result['flag']==0):

        st=db.collection('users').document(id).get()
        dat=st.to_dict()
        targ=dat['topic']
        tar=df.loc[df["Subcategory"]==targ]
        res=tar.values.tolist()

    if(result['flag']==1):

        pathc="users/"+id
        tot=[]
        st=db.collection('users').document(id+"enroll").get()
        so=st.to_dict()
        l=[]
        ans=[]
        for i in so:
            k=so[i]['name']
        l.append(k)
        for c in l:
            arr=np.arange(200)
            list_of_all_titles=df['Name'].tolist()
            find_close_match=difflib.get_close_matches(c,list_of_all_titles)
            close_match=find_close_match[0]
            index_of_the_movie=int(df[df.Name==close_match]['index'].values[0])
            similarity_score=list(enumerate(similarity[index_of_the_movie]))
            sorted_similar_movies=sorted(similarity_score,key= lambda x:x[1], reverse=True)
            i=1
            
            
            lst=[]
            lk=[]
            for course in sorted_similar_movies:
                index=course[0]
                title_from_index=df[df.index==index]['Name'].values[0]
                if i<3:
                    lst.append(title_from_index)
                  
                    i+=1
            tot.append(lst)
        res=[]
        for j in tot:
            for y in j:
                index1=int(df[df.Name==y]['index'].values[0])
                jk=df.loc[index1]
                res.append(jk)


    return render_template('index.html',data=res)


@app.route("/", methods =['GET', 'POST'])
def login():
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        try:
            user=auth.sign_in_with_email_and_password(email,password)
            session["user"]=user
        except:
            error=1
            return redirect(url_for('login'))
        return redirect('home')

    return  render_template("login.html")



@app.route("/single_course",methods=['GET','POST'])
def single_course():
    name=request.form['name']
    res1=df.loc[df["Name"]==name]
    sub=request.form['sub']
    sml=df.loc[df["Subcategory"]==sub]
    res=res1.values.tolist()
    res2=sml.values.tolist()
    res2=res2[:6]
    return  render_template("signlec.html",data=res,sm=res2)



@app.route("/signup",methods=['GET','POST'])
def signup():
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        user=auth.create_user_with_email_and_password(email,password)
        session["user"]=user

        return redirect('update')
    return render_template('signup.html',m="4444444444")

@app.route('/update',methods=['GET','POST'])
def update():
    user=session["user"]
    id=user["localId"]




    if request.method == 'POST' and 'Name' in request.form and 'Surname' in request.form:
        name=request.form['Name']
        lastname=request.form['Surname']
       
        address1=request.form['Address']
       
       
        state=request.form['State']
        
        email=user['email']
        education=request.form['education']
        country=request.form['country']
        
        topic=request.form['topic']
        yes=db.collection('users').document(id)
        file=request.files['file']
        data = {'name':name,'lastname':lastname,'address1':address1,'state':state,'email':email,'education':education,'country':country,'topic':topic}
        doc_ref =db.collection('users').document(id)
        doc = doc_ref.get()
        if doc.exists:
            yes.update(data)
        else:
            yes.set(data)
            sdata={'flag':0}
            db.collection('users').document(id).update(sdata)
    

        pathc="users/"+id
        storage.child(pathc).put(file)
        return redirect('home')



            
    return render_template('update.html',user=user)



@app.route("/logout")
def logout():
    session.pop("user",None)
    return redirect("/")

@app.route("/profile")
def profile():
    user=session["user"]
    id=user["localId"]
    pathc="users/"+id
    source=db.collection('users').document(id).get()
    data=source.to_dict()
    photo=storage.child(pathc).get_url(None)
    user=session["user"]
    id=user["localId"]
    pathc="users/"+id
    tot=[]
    st=db.collection('users').document(id+"enroll").get()
    so=st.to_dict()
    l=[]
    ans=[]
    for i in so:
        k=so[i]['name']
        l.append(k)
    for c in l:
            arr=np.arange(200)
            list_of_all_titles=df['Name'].tolist()
            find_close_match=difflib.get_close_matches(c,list_of_all_titles)
            close_match=find_close_match[0]
            index_of_the_movie=int(df[df.Name==close_match]['index'].values[0])
            similarity_score=list(enumerate(similarity[index_of_the_movie]))
            sorted_similar_movies=sorted(similarity_score,key= lambda x:x[1], reverse=True)
            i=1
            
            
            lst=[]
            lk=[]
            for course in sorted_similar_movies:
                index=course[0]
                title_from_index=df[df.index==index]['Name'].values[0]
                if i<3:
                    lst.append(title_from_index)
                  
                    i+=1
            tot.append(lst)
    res=[]
    for j in tot:
        for y in j:
            index1=int(df[df.Name==y]['index'].values[0])
            jk=df.loc[index1]
            res.append(jk)

    return render_template("profile.html",photo=photo,data=data,ml=res)

@app.route("/course")
def course():
    res=df.values.tolist()
    return render_template('course.html',res=res)



@app.route('/select', methods=['GET', 'POST'])
def select():
    if request.method == 'POST' and 'name' in request.form:

        user = session["user"]
        id = user["localId"]
        r = random.randint(0, 100000000000000000000)
        ran = str(r)
        rat=request.form['age']

        yes = db.collection('users').document(id + "enroll")
        ans1 = request.form['name']
        ans = {'name': ans1,'rate':rat}
        data = {ran: ans}

        all = db.collection('alldb').document("all")
        aldata = {'name': ans1, 'userid': id,'rate':rat}
        alldata = {ran: aldata}
        
        doc_ref = db.collection('users').document(id + "enroll")
        doc = doc_ref.get()
        if doc.exists:
            yes.update(data)
        else:
            yes.set(data)
        alldoc_ref = db.collection('alldb').document("all")
        alldoc = alldoc_ref.get()
        if alldoc.exists:
            all.update(alldata)
        else:
            all.set(alldata)
        

    return render_template('1.html')



@app.route('/courseex', methods=['GET', 'POST'])
def courseex():
    index=request.form['index']
    Float = float(index) 
    indx=int(Float)
    user = session["user"]
    id = user["localId"]
    sdata={'flag':1}
    
    relt=db.collection('users').document(id).get()
    result=relt.to_dict()
    if(result['flag']==0):
        db.collection('users').document(id).update(sdata)
            
    r = random.randint(0, 100000000000000000000)
    ran = str(r)

    yes = db.collection('users').document(id + "enroll")
    ans1 = request.form['name']
    ans = {'name': ans1}
    data = {ran: ans}

    aldata = {'name': ans1, 'userid': id}
    alldata = {ran: aldata}
        
    doc_ref = db.collection('users').document(id + "enroll")
    doc = doc_ref.get()
    if doc.exists:
        yes.update(data)
    else:
        yes.set(data)
    
    
    return render_template('brifc.html',data=ans1,index=indx)

@app.route('/thankyou', methods=['GET', 'POST'])
def thankyou():
    ans=request.form['name']
    rat=request.form['rate']
    scot=request.form['data']
    kk='[{"feed": "positive", "text": "'+ans+'" },{  "feed": "jkjkj",   "text": "good" }]'
    TweetDataSubset=pd.read_json(kk)
    nltk.download("stopwords")
    stop_words=set(stopwords.words("english"))
    wordnet = WordNetLemmatizer()
    def text_preproc(x):
        x=''.join([word for word in x.split(' ')if word not in stop_words])
        x=x.encode('ascii','ignore').decode()
        x=re.sub(r'https*\S+',' ',x)
        x=re.sub(r'@\S+',' ',x)
        x=re.sub(r'#\s+',' ',x)
        x=re.sub(r'\'\w+','',x)
        x=re.sub('[%s]'% re.escape(string.punctuation),'',x)
        x=re.sub(r'\w*\d+\w*','',x)
        x=re.sub(r'\s{2,}','',x)
        return x
    TweetDataSubset['clean_text']=TweetDataSubset.text.apply(text_preproc)
    x_train,x_test,y_train,y_test=train_test_split(TweetDataSubset["clean_text"],TweetDataSubset["feed"],test_size=0.1,shuffle=True)
    tfidf_vectorizer=TfidfVectorizer(use_idf=True,max_features=100)
    x_train_vectors_tfidf=tfidf_vectorizer.fit_transform(x_train)
    x_test_vectors_tfidf=tfidf_vectorizer.transform(x_test)
    lr_tfidf=MultinomialNB()
    lr_tfidf.fit(x_train_vectors_tfidf,y_train)
    y_predict=lr_tfidf.predict(x_test_vectors_tfidf)
    y_prob=lr_tfidf.predict_proba(x_test_vectors_tfidf)[:]
    Counter("".join(TweetDataSubset['clean_text']).split()).most_common(100)
    Positive_pattern_1=r"thanks"
    Positive_pattern_2=r"thank"
    Positive_pattern_3=r"good"
    Positive_pattern_4=r"well"
    Positive_pattern_5=r"super"
    Positive_pattern_6=r"awesome"
    Positive_pattern_7=r"satisfied"
    Positive_pattern_8=r"great"
    sco=int(scot)

    Postive_Pattern_List=[Positive_pattern_1,Positive_pattern_2,Positive_pattern_3,Positive_pattern_4,Positive_pattern_5,Positive_pattern_6,Positive_pattern_7,Positive_pattern_8]

    Positive_Complex_Pattern=re.compile('|'.join(['(%s)'% i for i in Postive_Pattern_List]),re.IGNORECASE)


    Negative_pattern_1=r"cancelled"
    Negative_pattern_2=r"delayed"
    Negative_pattern_3=r"trying"
    Negative_pattern_4=r"please"
    Negative_pattern_5=r"wait"
    Negative_pattern_6=r"worst"
    Negative_pattern_7=r"bad"
    Negative_pattern_8=r"tough"
    Negative_Pattern_List=[Negative_pattern_1,Negative_pattern_2,Negative_pattern_3,Negative_pattern_4,Negative_pattern_5,Negative_pattern_6,Negative_pattern_7,Negative_pattern_8]

    Negative_Complex_Pattern=re.compile('|'.join(['(%s)'% i for i in Negative_Pattern_List]),re.IGNORECASE)

    TweetDataSubset["Negative_Sentiment_Flag"]=TweetDataSubset["clean_text"].apply(lambda x:1 if(len(re.findall(Negative_Complex_Pattern,x))>0) else 0)

    TweetDataSubset["Positive_Sentiment_Flag"]=TweetDataSubset["clean_text"].apply(lambda x:1 if(len(re.findall(Positive_Complex_Pattern,x))>0) else 0)

    res=TweetDataSubset.values.tolist()
    if(sco > 3):
        boot=1
    else:
        boot=0
    a=res[0][3]
    b=res[0][4]
    p2=b-a
    if(p2 >0 ):
        feeds=1
    else:
        feeds=0
    raat=int(rat)
    if(feeds == 1):
        if(raat == 5):
            result=raat
        elif(raat == 4):
            result=raat+0.5
        else:
            result=raat+0.5+boot
    else:
        if(raat == 0):
            result=raat+boot
        elif(raat==1):
            result=raat-0.5+boot
        else:
            result=raat-0.5+boot

    
    index=request.form['index']
    indx=int(index)
    pgk=df.loc[indx]
    sub=pgk['Name']
    user = session["user"]
    id = user["localId"]
    r = random.randint(0, 100000000000000000000)
    ran = str(r)
        
    all = db.collection('alldb').document("all")
    ext = db.collection('inv').document(id)
    alldata = {'userid': id,'id':indx,'rate-level':rat,'sub':sub,'score':sco,'rate':result}
    alldata = {ran: alldata}
        
    alldoc_ref = db.collection('alldb').document("all")
    alldoc = alldoc_ref.get()
    adoc_ref = db.collection('inv').document(id)
    tdoc = adoc_ref.get()
   
    if alldoc.exists:
        all.update(alldata)
    else:
        all.set(alldata)
    if tdoc.exists:
        ext.update(alldata)
    else:
        ext.set(alldata)
   
    return render_template('thank.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    data=request.form['data']
    ans2=request.form['index']
    index=int(ans2)
    tmp=df.loc[index]


    return render_template('feedback.html',data=data,index=index)

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    
    index=request.form['index']
    indx=int(index)
    sub=df.loc[indx]
    return render_template('quiz.html',data=sub,index=index)

@app.route('/score', methods=['GET', 'POST'])
def score():
    ans1=request.form['fav_language']
    ans3=request.form['db']
    ans2=request.form['age']
    index=request.form['index']
    indx=int(index)
    sub=df.loc[indx]
    a=int(ans1)
    b=int(ans2)
    c=int(ans3)
    res=a+b+c
    return render_template('result.html',data=res,name=sub,index=indx)

@app.route('/1')
def hi():

    user=session["user"]
    id=user["localId"]
    pathc="users/"+id
    tot=[]
    st=db.collection('inv').document(id).get()
    so = st.to_dict()
    l=[]
    for i in so:

        l.append(so[i])
    k=json.dumps(l)
    input= pd.read_json(json.dumps(l))
    input=input.drop('score',1)
    input=input.drop('rate-level',1)
    input=input.drop('userid',1)



    stt=db.collection('alldb').document('all').get()
    soo = stt.to_dict()
    ll=[]
    for i in soo:

        ll.append(soo[i])
    k1=json.dumps(ll)
    rating = pd.read_json(json.dumps(ll))
    rating=rating.drop('score',1)
    rating=rating.drop('rate-level',1)

    usersubset=rating[rating['id'].isin(input['id'].tolist())]
    usersubsetGroup=usersubset.groupby(['userid'])
    usersubsetGroup=sorted(usersubsetGroup, key=lambda x: len(x[1]), reverse=True)
    usersubsetGroup =usersubsetGroup[0:100]
    pearsonCorrelationDict={}
    for name,group in usersubsetGroup:
        group=group.sort_values(by='id')
        input=input.sort_values(by='id')
    
        nRatings=len(group)
        temp_df=input[input['id'].isin(group['id'].tolist())]
    
        tempRatingList=temp_df['rate'].tolist()
    
        tempGroupList=group['rate'].tolist()
    
        Sxx= sum([i**2 for i in tempRatingList])-pow(sum(tempRatingList),2)/float(nRatings)
        Syy=sum([i**2 for i in tempGroupList])-pow(sum(tempGroupList),2)/float(nRatings)
        Sxy=sum(i*j for i,j in zip(tempRatingList,tempGroupList))-sum(tempRatingList)*sum(tempGroupList)
    
        if Sxx != 0  and Syy !=0:
            pearsonCorrelationDict[name]=Sxy/np.sqrt(Sxx*Syy)
        else:
            pearsonCorrelationDict[name]=0
    pearsonCorrelationDict.items()
    pearsonDF=pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    pearsonDF.columns=['similarityIndex']
    pearsonDF['userid']=pearsonDF.index
    pearsonDF.index=range(len(pearsonDF))
    topUsers=pearsonDF.sort_values(by='similarityIndex',ascending=False)[0:50]
    topUsersRating=topUsers.merge(rating,left_on='userid',right_on='userid',how='inner')
    topUsersRating['weightRating']=topUsersRating['similarityIndex']*topUsersRating['rate']
    temptopUsersRating=topUsersRating.groupby('id').sum()[['similarityIndex','weightRating']]
    temptopUsersRating.columns=['sum_similarityIndex','sum_weightRating']
    recommendation_df=pd.DataFrame()

    recommendation_df['weighted average recommendation score']=temptopUsersRating['sum_weightRating']/temptopUsersRating['sum_similarityIndex']
    recommendation_df['id']=temptopUsersRating.index

    recommendation_df=recommendation_df.sort_values(by='weighted average recommendation score',ascending=False)

    kk=rating.loc[rating['id'].isin(recommendation_df['id'].tolist())]











    return render_template('1.html',data=kk)






if __name__ == "__main__":
    app.run(debug=True)