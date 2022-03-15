from datetime import date, datetime
from pydoc import doc
import randomgit commit -m "first commit"
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

from flask import Flask , render_template, request, redirect, url_for, session, flash
from flask import Flask, jsonify, request
import numpy as np
import  pyrebase

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import csv


df = pd.read_csv('course.csv')
selected_features=['Name','Type','Category','Course_organization','Subcategory']

for feature in selected_features:
    df[feature]=df[feature].fillna('')

combined_features=df['Name']+''+df['Type']+''+df['Category']+''+df['Course_organization']+''+df['Subcategory']
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
    st=db.collection('users').document(id).get()
    data=st.to_dict()
    return render_template('index.html',data=data)


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


@app.route("/dashboard",methods=['GET','POST'])
def dashboard():
    return render_template('student-dashboard.html')


@app.route("/single_course")
def single_course():
    return  render_template("page-courses-computer-technologies.html")
@app.route("/contact")
def cou():
    return render_template('page-contact-style3.html')


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
        
        file=request.files['file']
        data = {'name':name,'lastname':lastname,'address1':address1,'state':state,'email':email,'education':education,'country':country,'topic':topic}
        db.collection('users').document(id).set(data)
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
    return render_template('course.html')

@app.route('/choose',methods=['GET','POST'])
def choose():
    if request.method == 'POST' and 'ans' in request.form:
        ans=request.form['ans']
        k=df.loc[df['Category']== ans]
        vals=k.values
        data=vals.tolist()
        return render_template('1.html',data=k)




    return render_template('choose.html')

@app.route('/eachcourse',methods=['GET','POST'])
def eachcourse():
    if request.method == 'POST' and 'ans1' in request.form:
        name=request.form['ans1']
        price=request.form['price']
        level=request.form['level']
        org=request.form['org']
        type=request.form['type']
        category=request.form['category']
        subcategory=request.form['subcategory']
        duration=request.form['duration']
        l=[name,price,level,org,type,category,subcategory,duration]
    return  render_template('eachcourse.html',data=l)


@app.route('/datascience', methods=['GET', 'POST'])
def datascience():
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

        return redirect('choose')


@app.route('/1')
def hi():

    user=session["user"]
    id=user["localId"]
    pathc="users/"+id
    tot=[]
    st=db.collection('alldb').document('all').get()
    so = st.to_dict()
    l=[]
    for i in so:

        l.append(so[i])
    k=json.dumps(l)



    return render_template('2.html',data=k)






if __name__ == "__main__":
    app.run(debug=True)