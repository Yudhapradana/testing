from app import app
import os, math, datetime
from flask import render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flaskext.mysql import MySQL
import csv, json, string
from xml.dom import minidom
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from flask import jsonify
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from flask_navigation import Navigation

# nav = Navigation(app)
mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'skripsi'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
tp = []
# nav.Bar('top', [
#     nav.Item('Home', 'index'),
#     nav.Item('News', 'news'),
# ])

@app.route('/')
@app.route('/index')
def index():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM testing INNER JOIN query ON testing.id_query=query.id ORDER BY isi ASC"
    cursor.execute(sql)
    k = cursor.fetchall()
    sql = "SELECT DISTINCT query.id, isi, typeir FROM testing INNER JOIN query ON testing.id_query=query.id ORDER BY isi ASC"
    cursor.execute(sql)
    query = cursor.fetchall()
    conn.close()
    return render_template('home.html', kluster=k, query=query)

@app.route('/news')
def news():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()
    return render_template('news.html',news=result)

@app.route('/createNews', methods=["POST"])
def createNews():
    title = request.form['title']
    desc = request.form['desc']
    source = request.form['source']
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "INSERT INTO news (judul, isi, sumber) VALUES (%s, %s, %s)"
    t = (title, desc, source)
    cursor.execute(sql, t)
    conn.commit()
    return redirect(url_for('news'))

@app.route('/updateNews', methods=["POST"])
def updateNews():
    id_news = request.form['id_news']
    title = request.form['utitle']
    desc = request.form['udesc']
    source = request.form['usource']
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "UPDATE news SET judul=%s, isi=%s, sumber=%s WHERE id_news=%s"
    t = (title, desc, source,id_news)
    cursor.execute(sql, t)
    conn.commit()
    return redirect(url_for('news'))

@app.route('/deleteNews/<string:id_news>', methods=['GET'])
def deleteNews(id_news):
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "DELETE FROM news WHERE id_news=%s"
    t = (id_news)
    cursor.execute(sql, t)
    conn.commit()
    return redirect(url_for('news'))

#upload
ALLOWED_EXTENSION=set(['csv','json','xml'])
app.config['UPLOAD_FOLDER'] = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION

@app.route('/importNews', methods=['GET','POST'])
def importNews():
    if request.method == 'POST':
        file = request.files['file']

        if 'file' not in request.files:
            return redirect(request.url)

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            ext = str(file.filename)
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            conn = mysql.connect()
            cursor = conn.cursor()

            if ext.rsplit('.',1)[1] == 'csv':
                csv_data = csv.reader(open(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                for row in csv_data:
                    t = str(row).strip('[]').strip("'")
                    b = t.rsplit(";")
                    sql = "INSERT INTO news (judul, isi, sumber) VALUES (%s, %s, %s)"
                    cursor.execute(sql, b)
            if ext.rsplit('.',1)[1] == 'json':
                json_data = json.load(open(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                for row in json_data:
                    judul = row['judul']
                    sumber = row['sumber']
                    isi = row['isi']
                    sql = "INSERT INTO news (judul, isi, sumber) VALUES (%s, %s, %s)"
                    t = (judul, isi, sumber)
                    cursor.execute(sql, t)
            if ext.rsplit('.', 1)[1] == 'xml':
                xml_data = minidom.parse(open(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                berita = xml_data.getElementsByTagName('berita')
                for row in berita:
                    sumber = row.getElementsByTagName('sumber')[0]
                    judul = row.getElementsByTagName('judul')[0]
                    isi = row.getElementsByTagName('isi')[0]
                    input1 = sumber.firstChild.data
                    input2 = judul.firstChild.data
                    input3 = isi.firstChild.data
                    sql = "INSERT INTO news (judul, isi, sumber) VALUES (%s, %s, %s)"
                    t = (input2, input3, input1)
                    cursor.execute(sql, t)
            conn.commit()
            cursor.close()
    return redirect(url_for('news'))

def stemming(listToStr):
    factorystem = StemmerFactory()
    stemmer = factorystem.create_stemmer()
    stemming = stemmer.stem(listToStr)
    return stemming

@app.route('/texpre')
def textpre():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    result = cursor.fetchall()
    if not tp:
        for row in result:
            print(row[0])
            # casefolding
            # mengubah text menjadi lowercase
            isi = str(row[2])
            lowercase = isi.lower()
            # menghapus tanda baca
            translator = str.maketrans('', '', string.punctuation)
            delpunctuation = lowercase.replace(".", " ").translate(translator)
            # Filtering
            # factory = StopWordRemoverFactory()
            # stopword = factory.create_stop_word_remover()
            # stop = stopword.remove(delpunctuation)
            stopwords = [line.rstrip() for line in open('uploads/stopword.txt')]
            cf = delpunctuation.split()
            stop = [a for a in cf if a not in stopwords]
            listToStr = ' '.join([str(elem) for elem in stop])
            # Stemming
            st = stemming(listToStr)
            #tokenizing
            token = st.split()
            if (row[0], row[1], row[3], delpunctuation, listToStr, st, token) not in tp:
                tp.append((row[0], row[1], row[3], delpunctuation, listToStr, st, token))
        sql = "SELECT * FROM word"
        cursor.execute(sql)
        word = cursor.fetchall()
        if not word:
            for row in tp:
                id_news = row[0]
                token = row[6]
                for row in token:
                    sql = "INSERT INTO word (word, id_news) VALUES(%s, %s)"
                    t = (row, id_news)
                    cursor.execute(sql, t)
        else:
            sql = "TRUNCATE word"
            cursor.execute(sql)
            for row in tp:
                id_news = row[0]
                token = row[6]
                for row in token:
                    sql = "INSERT INTO word (word, id_news) VALUES(%s, %s)"
                    t = (row, id_news)
                    cursor.execute(sql, t)
        conn.commit()
        sql = "SELECT * FROM wordset"
        cursor.execute(sql)
        wordset = cursor.fetchall()
        if not wordset:
            sql = "SELECT DISTINCT word FROM word"
            cursor.execute(sql)
            worddistinct = cursor.fetchall()
            for row in worddistinct:
                sql = "INSERT INTO wordset (wordset) VALUES(%s)"
                t = (row)
                cursor.execute(sql, t)
        else:
            sql = "TRUNCATE wordset"
            cursor.execute(sql)
            sql = "SELECT DISTINCT word FROM word"
            cursor.execute(sql)
            worddistinct = cursor.fetchall()
            for row in worddistinct:
                sql = "INSERT INTO wordset (wordset) VALUES(%s)"
                t = (row)
                cursor.execute(sql, t)
    conn.commit()
    cursor.close()
    return render_template('textpreprocessing.html', textpre=tp)

@app.route('/getNews')
def getNews():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()

    res = []
    for row in result:
        res.append({ "id" : row[0],"judul" : row[1],})

    data = {}
    data["data"] = res
    return jsonify(data)

@app.route('/tfidf')
def tfidf():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    with open('tfidf.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    if len(news) != len(json_object):
        sql = "SELECT * FROM word"
        cursor.execute(sql)
        result = cursor.fetchall()
        word = []
        wordset = []
        wd= []
        idf = []
        finaltfidf = []
        for row in result:
            word.append(row[1])
        for row in word:
            if row not in wordset:
                wordset.append(row)
        listberita = []
        for row in news:
            listword = []
            id_news = row[0]
            # print(id_news)
            sql = "SELECT * FROM word WHERE id_news=%s"
            t = (id_news)
            cursor.execute(sql, t)
            word = cursor.fetchall()
            for row in word:
                w = str(row[1])
                listword.append(w)
            listberita.append((id_news, listword))
        for row in listberita:
            id_news = row[0]
            # print("listberita"+str(id_news))
            isi = row[1]
            worddict = dict.fromkeys(wordset, 0)
            for word in isi:
                if word in worddict:
                    worddict[word] += 1
            wd.append((id_news, worddict))
            idf.append(worddict)
        idfword = computeIDF(idf)
        for row in wd:
            wordtfidf = computeTFIDF(row[1], idfword)
            finaltfidf.append((row[0], wordtfidf))
        all = []
        for row in finaltfidf:
            idnews = row[0]
            tfidf = row[1]
            tfidf.update({'idnews' : idnews})
            all.append(tfidf)
        # Serializing json
        with open("tfidf.json", "w") as outfile:
            json.dump(all, outfile)

        with open('tfidf.json', 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
    conn.commit()
    cursor.close()
    return render_template('tfidf.html', tfidf=news, list=json_object)

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

def computeIDF(docList):
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        if val < 1:
            idfDict[word] = 0
        else:
            idfDict[word] = (float(math.log10(N/float(val))))+1
    return idfDict

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

@app.route('/doc2vec')
def doc2vec():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    listberita = []
    idnews = []
    title = []
    for row in news:
        listword = []
        id_news = row[0]
        judul = row[1]
        sql = "SELECT * FROM word WHERE id_news=%s"
        t = (id_news)
        cursor.execute(sql, t)
        word = cursor.fetchall()
        for row in word:
            w = str(row[1])
            listword.append(w)
        listberita.append((id_news, listword))
        idnews.append(id_news)
        title.append(judul)
    data = []
    for row in listberita:
        sentence = ""
        word = row[1]
        for row in word:
            sentence+=row+" "
        data.append(str(sentence))
    tagged_data = [TaggedDocument(words=_d.split(), tags=[str(i)]) for i, _d in enumerate(data)]
    model = Doc2Vec(tagged_data)
    model.save("dv2.model")
    # print(len(model.infer_vector(["4g", "lte"])))
    N = len(model.docvecs)
    vecdoc = []
    for row in range(N):
        vecdoc.append(model.docvecs[row])
    joinidnewsvecdoc = np.array([idnews, title, vecdoc])
    joinidnewsvecdoc_transpose = joinidnewsvecdoc.transpose()
    # array = np.array(vecdoc)
    # kmeans = MiniBatchKMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None).fit(array)
    return render_template('doc2vec.html', result=joinidnewsvecdoc_transpose)

@app.route('/getKmeansTfidf')
def getKmeansTfidf():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM cluster_kmeans WHERE type='tfidf' ORDER BY id ASC"
    cursor.execute(sql)
    kmeans = cursor.fetchall()
    return render_template('kmeans.html', kmeans=kmeans)

@app.route('/checkCluster')
def checkcluster():
    matrix = []
    with open('tfidf.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        for row in json_object:
            listval = []
            del row['idnews']
            for word,val in row.items():
                listval.append(float(val))
            matrix.append(listval)
    array = np.array(matrix)
    wcss = []
    for i in range(1, 500):
        print(i)
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=1)
        kmeans.fit(array)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(16, 8))
    plt.plot(range(1,500), wcss, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    return redirect(url_for('getKmeansTfidf'))

@app.route('/getClusterTfidf', methods=["POST"])
def getClusterTfidf():
    conn = mysql.connect()
    cursor = conn.cursor()
    matrix = []
    with open('tfidf.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        for row in json_object:
            listval = []
            del row['idnews']
            for word, val in row.items():
                listval.append(float(val))
            matrix.append(listval)
    array = np.array(matrix)
    total = request.form['total']
    # wcss = []
    # for i in range(1, 4):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    #     kmeans.fit(array)
    #     wcss.append(kmeans.inertia_)
    # # plt.plot(range(1, 4), wcss)
    # # plt.title('Metode Elbow')
    # # plt.xlabel('Jumlah cluster')
    # # plt.ylabel('WCSS')
    # # plt.show()
    print("kmeans")
    kmeans = KMeans(n_clusters=int(total), init='k-means++', random_state=1)
    kmeans.fit(array)
    resultcluster = kmeans.labels_
    print(resultcluster)
    centroid = kmeans.cluster_centers_
    c = []
    for row in resultcluster:
        c.append("Cluster "+str(row))
    c2 = []
    for row in c:
        print(row)
        if row not in c2:
            c2.append(row)
    type = "tfidf"
    for row in c2:
        # print("c2"+str(row))
        cluster = row
        datakmeans = "SELECT cluster, total FROM cluster_kmeans WHERE cluster=%s AND total=%s AND type=%s"
        t = (cluster, total, type)
        cursor.execute(datakmeans, t)
        result = cursor.fetchall()
        if not result:
            sql = "INSERT INTO cluster_kmeans(cluster ,total, type) VALUES(%s, %s, %s)"
            t = (cluster, total, type)
            cursor.execute(sql, t)
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    id_news = []
    for row in news:
        id_news.append(row[0])
    idkmeans=[]
    for row in c:
        clus = row
        print(clus)
        sql = "SELECT id FROM cluster_kmeans WHERE cluster=%s AND total=%s AND type=%s"
        t = (clus, total, type)
        cursor.execute(sql, t)
        id = cursor.fetchone()
        id = str(id).replace("(", "").replace(")", "").replace(",", "")
        print(id)
        idkmeans.append(id)
    joinidnewsidkmeans = np.array([id_news, idkmeans])
    joinidnewsidkmeans_transpose = joinidnewsidkmeans.transpose()
    # print("join")
    for row in joinidnewsidkmeans_transpose:
        text = str(row).replace("['", "").replace("' '", " ").replace("']", "").split(" ")
        id = text[0]
        # print("join"+str(id))
        idc = text[1]
        dataresultkmeans = "SELECT total FROM result_kmeans INNER JOIN cluster_kmeans on result_kmeans.id_cluster=cluster_kmeans.id WHERE result_kmeans.id_news=%s AND total=%s AND type=%s"
        t = (id, total, type)
        cursor.execute(dataresultkmeans, t)
        result = cursor.fetchall()
        totalset = []
        for r in result:
            r = str(r).replace("(", "").replace(")", "").replace(",", "")
            if r not in totalset:
                totalset.append(r)
        if not totalset:
            sql = "INSERT INTO result_kmeans(id_cluster, id_news) VALUES(%s, %s)"
            t = (idc, id)
            cursor.execute(sql, t)
    sql = "SELECT * FROM word"
    cursor.execute(sql)
    result = cursor.fetchall()
    wordset = []
    for row in result:
        if row[1] not in wordset:
            wordset.append(row[1])
    i = 0
    all = []
    # print("centroid")
    for row in centroid:
        cluster = "Cluster "+str(i)
        # print(cluster)
        joincentroidwordset = np.array([wordset, row])
        transpose = joincentroidwordset.transpose()
        worddict = dict.fromkeys(wordset, 0)
        for rows in transpose:
            worddict[rows[0]] = float(rows[1])
        worddict.update({'idcluster' : cluster})
        all.append(worddict)
        i+=1
    print("save")
    with open("centroidkmeans"+total+".json", "w") as outfile:
        json.dump(all, outfile)
    conn.commit()
    cursor.close()
    return redirect(url_for('getKmeansTfidf'))

@app.route('/getKmeansDoc2vec')
def getKmeansDoc2vec():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM cluster_kmeans WHERE type='doc2vec' ORDER BY id ASC"
    cursor.execute(sql)
    kmeans = cursor.fetchall()
    return render_template('kmeans2.html', kmeans=kmeans)

@app.route('/getClusterDoc2vec', methods=["POST"])
def getClusterDoc2vec():
    conn = mysql.connect()
    cursor = conn.cursor()
    total = request.form['total']
    model = Doc2Vec.load("dv2.model")
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    # listberita = []
    idnews = []
    for row in news:
        # listword = []
        id_news = row[0]
        # sql = "SELECT * FROM word WHERE id_news=%s"
        # t = (id_news)
        # cursor.execute(sql, t)
        # word = cursor.fetchall()
        # for row in word:
        #     w = str(row[1])
        #     listword.append(w)
        # listberita.append((id_news, listword))
        idnews.append(id_news)
    # data = []
    # for row in listberita:
    #     sentence = ""
    #     word = row[1]
    #     for row in word:
    #         sentence += row + " "
    #     data.append(str(sentence))
    # tagged_data = [TaggedDocument(words=_d.split(), tags=[str(i)]) for i, _d in enumerate(data)]
    # model = Doc2Vec(tagged_data)
    # model.save("dv2.model")
    # print(len(model.infer_vector(["4g", "lte"])))
    N = len(model.docvecs)
    vecdoc = []
    for row in range(N):
        vecdoc.append(model.docvecs[row])
    array = np.array(vecdoc)
    kmeans = KMeans(n_clusters=int(total), init='k-means++', random_state=1)
    kmeans.fit(array)
    resultcluster = kmeans.labels_
    centroidcluster = kmeans.cluster_centers_
    c = []
    for row in resultcluster:
        c.append(("Cluster " + str(row), 0))
    c2 = []
    for row in c:
        if row not in c2:
            c2.append(row)
    type = "doc2vec"
    for row in c2:
        cluster = row[0]
        datakmeans = "SELECT cluster, total FROM cluster_kmeans WHERE cluster=%s AND total=%s AND type=%s"
        t = (cluster, total, type)
        cursor.execute(datakmeans, t)
        result = cursor.fetchall()
        if not result:
            sql = "INSERT INTO cluster_kmeans(cluster ,total, type) VALUES(%s, %s, %s)"
            t = (cluster, total, type)
            cursor.execute(sql, t)
    idkmeans = []
    for row in c:
        clus = row[0]
        sql = "SELECT id FROM cluster_kmeans WHERE cluster=%s AND total=%s AND type=%s"
        t = (clus, total, type)
        cursor.execute(sql, t)
        id = cursor.fetchone()
        id = str(id).replace("(", "").replace(")", "").replace(",", "")
        idkmeans.append(id)
    joinidnewsidkmeans = np.array([idnews, idkmeans])
    joinidnewsidkmeans_transpose = joinidnewsidkmeans.transpose()
    for row in joinidnewsidkmeans_transpose:
        text = str(row).replace("['", "").replace("' '", " ").replace("']", "").split(" ")
        id = text[0]
        idc = text[1]
        dataresultkmeans = "SELECT total FROM result_kmeans INNER JOIN cluster_kmeans on result_kmeans.id_cluster=cluster_kmeans.id WHERE result_kmeans.id_news=%s AND total=%s AND type=%s"
        t = (id, total, type)
        cursor.execute(dataresultkmeans, t)
        result = cursor.fetchall()
        totalset = []
        for r in result:
            r = str(r).replace("(", "").replace(")", "").replace(",", "")
            if r not in totalset:
                totalset.append(r)
        if not totalset:
            sql = "INSERT INTO result_kmeans(id_cluster, id_news) VALUES(%s, %s)"
            t = (idc, id)
            cursor.execute(sql, t)
    i = 0
    all = []
    print("centroid")
    for row in centroidcluster:
        print(len(row))
        cluster = "Cluster " + str(i)
        print(cluster)
        dict = {
            'idcluster' : cluster,
            'valcentroid' : str(row)
        }
        all.append(dict)
        i += 1
    print("save")
    with open("centroidkmeansdoc2vec"+total+".json", "w") as outfile:
        json.dump(all, outfile)
    conn.commit()
    conn.close()
    return redirect(url_for('getKmeansDoc2vec'))

@app.route('/getClusterNews')
def getClusterNews():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT DISTINCT cluster FROM cluster_kmeans"
    cursor.execute(sql)
    cluster = cursor.fetchall()
    sql = "SELECT DISTINCT total FROM cluster_kmeans"
    cursor.execute(sql)
    total = cursor.fetchall()
    cursor.close()
    return render_template('clusternews.html', cluster=cluster, total=total)

@app.route('/searchClusterNews',methods=['GET','POST'])
def searchClusterNews():
    conn = mysql.connect()
    cursor = conn.cursor()
    cluster = request.form['cluster']
    total = request.form['total']
    type = request.form['type']
    sql = "SELECT k.total, k.cluster, n.judul FROM `result_kmeans` INNER JOIN cluster_kmeans as k on k.id=result_kmeans.id_cluster INNER JOIN news as n on n.id_news=result_kmeans.id_news WHERE k.cluster=%s AND k.total=%s AND k.type=%s ORDER BY k.cluster ASC"
    t = (cluster, total, type)
    cursor.execute(sql, t)
    result = cursor.fetchall()
    sql = "SELECT DISTINCT cluster FROM cluster_kmeans"
    cursor.execute(sql)
    cluster = cursor.fetchall()
    sql = "SELECT DISTINCT total FROM cluster_kmeans"
    cursor.execute(sql)
    total = cursor.fetchall()
    cursor.close()
    return render_template('clusternews.html', result=result, cluster=cluster, total=total)

@app.route('/ir_tfidf_kmeans')
def ir_tfidf_kmeans():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT DISTINCT total FROM cluster_kmeans WHERE type='tfidf'"
    cursor.execute(sql)
    selecttotal = cursor.fetchall()
    return render_template('irs_tfidf_kmeans.html', total=selecttotal)

@app.route('/processTfidfKmeans', methods=['GET','POST'])
def processTfidfKmeans():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT DISTINCT total FROM cluster_kmeans WHERE type='tfidf'"
    cursor.execute(sql)
    selecttotal = cursor.fetchall()
    query = str(request.form['query'])
    total = request.form['total']
    threshold = request.form['threshold']
    a = datetime.datetime.now().replace(microsecond=0)
    #text preprocessing
    lowercase = query.lower()
    translator = str.maketrans('', '', string.punctuation)
    delpunctuation = lowercase.replace(".", " ").translate(translator)
    stopwords = [line.rstrip() for line in open('uploads/stopword.txt')]
    cf = delpunctuation.split()
    stop = [a for a in cf if a not in stopwords]
    listToStr = ' '.join([str(elem) for elem in stop])
    st = stemming(listToStr)
    token = st.split()
    # end
    dataquery = []
    dataquery.append(('q', token))
    word = []
    wordset = []
    sql = "SELECT * FROM word"
    cursor.execute(sql)
    result = cursor.fetchall()
    for row in result:
        word.append(row[1])
    for row in word:
        if row not in wordset:
            wordset.append(row)
    listcentroid = []
    with open("centroidkmeans"+total+".json", "r") as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        for row in json_object:
            cluster = row['idcluster']
            del row['idcluster']
            listcentroid.append((cluster, row))
    for row in dataquery:
        token = row[1]
        worddict = dict.fromkeys(wordset, 0)
        for rows in token:
            if rows in worddict:
                worddict[rows]+=1
        listcentroid.append(('q', worddict))
    listworddict = []
    for row in listcentroid:
        worddict = row[1]
        listworddict.append(worddict)
    idfword = computeIDF(listworddict)
    qtfidf = []
    j = 0
    finaltfidf = []
    for row in listcentroid:
        wordtfidf = computeTFIDF(row[1], idfword)
        if row[0] == 'q':
            qtfidf.append((row[0], wordtfidf))
        if j == (len(listcentroid)-1):
            listcentroid.pop(j)
        j+=1
        # finaltfidf.append((row[0], wordtfidf))
    for row in qtfidf:
        listcentroid.append(row)
    computeScalar = computeCrossScalar(listcentroid)
    sumComputeScalar = computeSumScalar(computeScalar)
    lovector = longvector(listcentroid)
    N = len(lovector)
    i = 0
    datalovector = []
    for row in lovector:
        if i == (N-1):
            qlovector = row
        else:
            datalovector.append(row)
        i+=1
    joinsumscalardatalovector = np.array([sumComputeScalar, datalovector])
    joinsumscalardatalovector_transpose = joinsumscalardatalovector.transpose()
    cos = []
    c = []
    i = 0
    for row in joinsumscalardatalovector_transpose:
        sumScalar = row[0]
        longvec = row[1]
        C = sumScalar/(qlovector*longvec)
        cos.append((i, C))
        c.append(C)
        i+=1
    print(c)
    print(cos)
    final = []
    for row in cos:
        cluster = "Cluster "+str(row[0])
        if row[1] > float(threshold):
            sql = "SELECT id FROM cluster_kmeans WHERE cluster=%s AND total=%s"
            t = (cluster, total)
            cursor.execute(sql, t)
            result = cursor.fetchall()
            result = str(result).replace("(", "").replace(")", "").replace(",", "")
            if result:
                sql = "SELECT news.id_news, news.judul, news.isi, news.sumber FROM result_kmeans INNER JOIN news ON result_kmeans.id_news=news.id_news WHERE id_cluster=%s"
                t = (result)
                cursor.execute(sql, t)
                result = cursor.fetchall()
                for rows in result:
                    final.append(rows)
    b = datetime.datetime.now().replace(microsecond=0)
    c = b - a
    c = c.seconds
    # recall
    sql = "SELECT id FROM query WHERE isi=%s"
    t = (query)
    cursor.execute(sql, t)
    idquery = cursor.fetchall()
    idquery = str(idquery).replace("(", "").replace(",", "").replace(")", "")
    sql = "SELECT * FROM query_news WHERE id_query=%s"
    t = (idquery)
    cursor.execute(sql, t)
    n1 = cursor.fetchall()
    print(n1)
    count = 0
    for row in final:
        id = row[0]
        for rows in n1:
            id2 = rows[1]
            if id == id2:
                count += 1
    print(len(final))
    print(count)
    recall = float(count / len(n1))
    precission = float(count / len(final))
    fscore = float(2 * (recall * precission) / (recall + precission))
    sql = "SELECT * FROM testing WHERE id_query=%s AND threshold=%s AND typeir=%s"
    t = (idquery, threshold, "tfidf")
    cursor.execute(sql, t)
    check = cursor.fetchall()
    print(check)
    if not check:
        sql = "INSERT INTO testing(id_query, threshold, relevan, n1, n2, time, recall, precission, fscore, typeir) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        t = (idquery, str(threshold), count, len(n1), len(final), c, recall, precission, fscore, "tfidf")
        cursor.execute(sql, t)
    else:
        sql = "UPDATE testing SET relevan=%s, n1=%s, n2=%s, time=%s, recall=%s, precission=%s, fscore=%s WHERE id_query=%s AND threshold=%s AND typeir=%s"
        t = (count, len(n1), len(final), c, recall, precission, fscore, idquery, str(threshold), "tfidf")
        cursor.execute(sql, t)
    conn.commit()
    return render_template('irs_tfidf_kmeans.html', total=selecttotal, result=final)

@app.route('/irs_doc2vec_kmeans')
def irs_doc2vec_kmeans():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT DISTINCT total FROM cluster_kmeans WHERE type='doc2vec'"
    cursor.execute(sql)
    selecttotal = cursor.fetchall()
    conn.close()
    return render_template('irs_doc2vec_kmeans.html', total=selecttotal)

@app.route('/processDoc2vecKmeans', methods=['GET','POST'])
def processDoc2vecKmeans():
    conn = mysql.connect()
    cursor = conn.cursor()
    model = Doc2Vec.load("dv2.model")
    query = str(request.form['query'])
    total = request.form['total']
    threshold = request.form['threshold']
    sql = "SELECT DISTINCT total FROM cluster_kmeans WHERE type='doc2vec'"
    cursor.execute(sql)
    selecttotal = cursor.fetchall()
    # N = len(model.docvecs)
    # vecdoc = []
    # for row in range(N):
    #     vecdoc.append(model.docvecs[row])
    # array = np.array(vecdoc)
    # kmeans = MiniBatchKMeans(n_clusters=int(total), init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0,
    #                          random_state=123)
    # kmeans.fit(array)
    # resultcluster = kmeans.labels_
    # centroidcluster = kmeans.cluster_centers_
    # c = []
    # for row in resultcluster:
    #     c.append("Cluster " + str(row))
    # sql = "SELECT * FROM news"
    # cursor.execute(sql)
    # news = cursor.fetchall()
    # idnews = []
    # for row in news:
    #     id = row[0]
    #     idnews.append(id)
    # idnewscluster = np.array([idnews, c])
    # idnewscluster_transpose = idnewscluster.transpose()
    # print(idnewscluster_transpose)
    #text preprocessing
    a = datetime.datetime.now().replace(microsecond=0)
    lowercase = query.lower()
    translator = str.maketrans('', '', string.punctuation)
    delpunctuation = lowercase.replace(".", " ").translate(translator)
    stopwords = [line.rstrip() for line in open('uploads/stopword.txt')]
    cf = delpunctuation.split()
    stop = [a for a in cf if a not in stopwords]
    listToStr = ' '.join([str(elem) for elem in stop])
    st = stemming(listToStr)
    token = st.split()
    # get vector
    vecquery = model.infer_vector(token, steps=20)
    # print(vecquery)
    vectorDocCluster = []
    #getcentroid
    centroidcluster = []
    with open("centroidkmeansdoc2vec"+total+".json", "r") as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        for row in json_object:
            cluster = row['idcluster']
            valcentroid = str(row['valcentroid'])
            valcentroid = valcentroid.replace("'", "").replace("[", "").replace("]", "").split()
            centroidcluster.append((cluster, valcentroid))
    for row in centroidcluster:
        vectorDocCluster.append(row[1])
    vectorDocCluster.append(vecquery)
    #computecrossscalar
    resultComputeCrossScalar = []
    valvec = []
    for row in vecquery:
        valvec.append(float(row))
    for row in centroidcluster:
        value = []
        centroid = row[1]
        array = np.array([valvec, centroid])
        array_transpose = array.transpose()
        for row2 in array_transpose:
            # if float(row2[0]) < 0:
            #     row2[0] = 0
            # if float(row2[1]) < 0:
            #     row2[1] = 0
            score = float(float(row2[0])*float(row2[1]))
            value.append(score)
        resultComputeCrossScalar.append(value)
    #computesumscalar
    sum = []
    for row in resultComputeCrossScalar:
        W = 0
        for rows in row:
            W+=float(rows)
        sum.append(W)
    #longvector
    lvector = []
    for row in vectorDocCluster:
        valpow= []
        vector = row
        for row1 in vector:
            score = math.pow(float(row1), 2)
            print(score)
            valpow.append(score)
        W = 0
        for row2 in valpow:
            W+=float(row2)
        akar = math.sqrt(W)
        lvector.append(akar)
    N = len(lvector)
    i = 0
    datalvector = []
    for row in lvector:
        if i==(N-1):
            qlvector = row
        else:
            datalvector.append(row)
            i+=1
    joinsumscalardatalvector = np.array([sum, datalvector])
    joinsumscalardatalvector_transpose = joinsumscalardatalvector.transpose()
    c = []
    cos = []
    i = 0
    for row in joinsumscalardatalvector_transpose:
        sume = row[0]
        lvectore = row[1]
        C = sume / (qlvector*lvectore)
        cos.append((i, C))
        c.append(C)
        i+=1
    # valuemax = max(c)
    # for row in cos:
    #     if row[1] == valuemax:
    #         choosecluster = "Cluster "+str(row[0])
    # finalnews = []
    # for row in idnewscluster_transpose:
    #     if row[1] == choosecluster:
    #         finalnews.append(row[0])
    # final = []
    # for row in finalnews:
    #     id = row
    #     sql = "SELECT * FROM news WHERE id_news=%s"
    #     t = (id)
    #     cursor.execute(sql, t)
    #     result = cursor.fetchall()
    #     for rows in result:
    #         final.append((rows[0], rows[1], rows[2], rows[3]))
    choosecluster = []
    print(cos)
    for row in cos:
        if row[1] > float(threshold):
            choosecluster.append("Cluster "+str(row[0]))
    print(choosecluster)
    print(len(choosecluster))
    finalnews = []
    sql = "SELECT id_news, cluster FROM result_kmeans INNER JOIN cluster_kmeans on result_kmeans.id_cluster=cluster_kmeans.id WHERE cluster_kmeans.total=%s AND cluster_kmeans.type=%s ORDER BY id_news ASC"
    t = (total, "doc2vec")
    cursor.execute(sql, t)
    idnewscluster_transpose = cursor.fetchall()
    for row in idnewscluster_transpose:
        if row[1] in choosecluster:
            finalnews.append(row[0])
    print(len(finalnews))
    final = []
    for row in finalnews:
        id = row
        sql = "SELECT * FROM news WHERE id_news=%s"
        t = (id)
        cursor.execute(sql, t)
        result = cursor.fetchall()
        for rows in result:
            final.append((rows[0], rows[1], rows[2], rows[3]))
    b = datetime.datetime.now().replace(microsecond=0)
    c = b - a
    c = c.seconds
    # recall
    sql = "SELECT id FROM query WHERE isi=%s"
    t = (query)
    cursor.execute(sql, t)
    idquery = cursor.fetchall()
    idquery = str(idquery).replace("(", "").replace(",", "").replace(")", "")
    sql = "SELECT * FROM query_news WHERE id_query=%s"
    t = (idquery)
    cursor.execute(sql, t)
    n1 = cursor.fetchall()
    print(n1)
    count = 0
    for row in final:
        id = row[0]
        for rows in n1:
            id2 = rows[1]
            if id == id2:
                count += 1
    print(len(final))
    print(count)
    recall = float(count / len(n1))
    precission = float(count / len(final))
    fscore = float(2 * (recall * precission) / (recall + precission))
    sql = "SELECT * FROM testing WHERE id_query=%s AND threshold=%s AND typeir=%s"
    t = (idquery, threshold, "doc2vec")
    cursor.execute(sql, t)
    check = cursor.fetchall()
    print(check)
    if not check:
        sql = "INSERT INTO testing(id_query, threshold, relevan, n1, n2, time, recall, precission, fscore, typeir) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        t = (idquery, str(threshold), count, len(n1), len(final), c, recall, precission, fscore, "doc2vec")
        cursor.execute(sql, t)
    else:
        sql = "UPDATE testing SET relevan=%s, n1=%s, n2=%s, time=%s, recall=%s, precission=%s, fscore=%s WHERE id_query=%s AND threshold=%s AND typeir=%s"
        t = (count, len(n1), len(final), c, recall, precission, fscore, idquery, str(threshold), "doc2vec")
        cursor.execute(sql, t)
    conn.commit()
    return render_template('irs_doc2vec_kmeans.html', data=final, total=selecttotal)

@app.route('/ir_nocluster')
def ir_nocluster():
    return render_template('irs_nocluster.html')

@app.route('/processNoCluster', methods=['GET','POST'])
def processNoCluster():
    conn = mysql.connect()
    cursor = conn.cursor()
    query = str(request.form['query'])
    threshold = str(request.form['threshold'])
    # text preprocessing
    a = datetime.datetime.now().replace(microsecond=0)
    lowercase = query.lower()
    translator = str.maketrans('', '', string.punctuation)
    delpunctuation = lowercase.replace(".", " ").translate(translator)
    stopwords = [line.rstrip() for line in open('uploads/stopword.txt')]
    cf = delpunctuation.split()
    stop = [a for a in cf if a not in stopwords]
    listToStr = ' '.join([str(elem) for elem in stop])
    st = stemming(listToStr)
    token = st.split()
    listberita = []
    wd = []
    word = []
    wordset = []
    idf = []
    finaltfidf = []
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    sql = "SELECT * FROM word"
    cursor.execute(sql)
    result = cursor.fetchall()
    for row in result:
        word.append(row[1])
    for row in word:
        if row not in wordset:
            wordset.append(row)
    for row in news:
        listword = []
        id_news = row[0]
        sql = "SELECT * FROM word WHERE id_news=%s"
        t = (id_news)
        cursor.execute(sql, t)
        word = cursor.fetchall()
        for row in word:
            w = str(row[1])
            listword.append(w)
        listberita.append((id_news, listword))
    listberita.append(("q", token))
    for row in listberita:
        id_news = row[0]
        isi = row[1]
        worddict = dict.fromkeys(wordset, 0)
        for word in isi:
            if word in worddict:
                worddict[word] += 1
        wd.append((id_news, worddict))
        idf.append(worddict)
    idfword = computeIDF(idf)
    for row in wd:
        wordtfidf = computeTFIDF(row[1], idfword)
        finaltfidf.append((row[0], wordtfidf))
    computeScalar = computeCrossScalar(finaltfidf)
    sumComputeScalar = computeSumScalar(computeScalar)
    lovector = longvector(finaltfidf)
    N = len(lovector)
    i = 0
    datalovector = []
    for row in lovector:
        if i == (N - 1):
            qlovector = row
        else:
            datalovector.append(row)
        i += 1
    joinsumscalardatalovector = np.array([sumComputeScalar, datalovector])
    joinsumscalardatalovector_transpose = joinsumscalardatalovector.transpose()
    c = []
    for row in joinsumscalardatalovector_transpose:
        sumScalar = row[0]
        longvec = row[1]
        C = sumScalar / (qlovector * longvec)
        c.append(C)
    sql = "SELECT id_news FROM news"
    cursor.execute(sql)
    idnews= cursor.fetchall()
    joinidnewsc = np.array([idnews, c])
    joinidnewsc_transpose = joinidnewsc.transpose()
    final = []
    print(joinidnewsc_transpose)
    for row in joinidnewsc_transpose:
        id = row[0]
        valuecosine = row[1]
        if valuecosine > float(threshold):
            print(valuecosine)
            sql = "SELECT * FROM news WHERE id_news=%s"
            t = (id)
            cursor.execute(sql, t)
            result = cursor.fetchall()
            for row in result:
                final.append((row[0], row[1], row[2], row[3]))
    b = datetime.datetime.now().replace(microsecond=0)
    c = b-a
    c = c.seconds
    #recall
    sql = "SELECT id FROM query WHERE isi=%s"
    t = (query)
    cursor.execute(sql, t)
    idquery = cursor.fetchall()
    idquery = str(idquery).replace("(", "").replace(",", "").replace(")", "")
    sql = "SELECT * FROM query_news WHERE id_query=%s"
    t = (idquery)
    cursor.execute(sql, t)
    n1 = cursor.fetchall()
    print(n1)
    count = 0
    for row in final:
        id = row[0]
        for rows in n1:
            id2 = rows[1]
            if id == id2:
                count += 1
    print(len(final))
    print(count)
    recall = float(count / len(n1))
    precission = float(count / len(final))
    fscore = float(2 * (recall * precission) / (recall + precission))
    sql = "SELECT * FROM testing WHERE id_query=%s AND threshold=%s AND typeir=%s"
    t = (idquery, threshold, "nocluster")
    cursor.execute(sql, t)
    check = cursor.fetchall()
    print(check)
    if not check:
        sql = "INSERT INTO testing(id_query, threshold, relevan, n1, n2, time, recall, precission, fscore, typeir) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        t = (idquery, str(threshold), count, len(n1), len(final), c, recall, precission, fscore, "nocluster")
        cursor.execute(sql, t)
    else:
        sql = "UPDATE testing SET relevan=%s, n1=%s, n2=%s, time=%s, recall=%s, precission=%s, fscore=%s WHERE id_query=%s AND threshold=%s AND typeir=%s"
        t = (count, len(n1), len(final), c, recall, precission, fscore, idquery, str(threshold), "nocluster")
        cursor.execute(sql, t)
    conn.commit()
    return render_template('irs_nocluster.html', result=final)

def computeCrossScalar(tfidf):
    N = len(tfidf)
    i = 0
    Q = []
    doc = []
    for row in tfidf:
        if i == (N-1):
            Q.append(row[1])
        else:
            doc.append(row[1])
        i+=1
    result = []
    for row in doc:
        wdDict = dict.fromkeys(doc[0].keys(), 0)
        for row2 in Q:
            for word, val in row2.items():
                wdDict[word]+=float(val)
        for word, val in row.items():
            wdDict[word]*=float(val)
        result.append(wdDict)
    return result

def computeSumScalar(computeScalar):
    sum = []
    for row in computeScalar:
        W = 0
        for word, val in row.items():
            W+=float(val)
        sum.append(W)
    return sum

def longvector(tfidf):
    result = []
    for row in tfidf:
        value = row[1]
        wdDict = dict.fromkeys(row[1].keys(), 0)
        for word, val in value.items():
            wdDict[word] = math.pow(val, 2)
        W = 0
        for word, val in wdDict.items():
            W+=float(val)
        akar = math.sqrt(W)
        result.append(akar)
    return result

@app.route('/test_tfidf')
def test_tfidf():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM testing WHERE typeir='tfidf' ORDER BY query ASC"
    cursor.execute(sql)
    result = cursor.fetchall()
    return render_template('test_tfidf.html', result=result)

@app.route('/test_doc2vec')
def test_doc2vec():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM testing WHERE typeir='doc2vec' ORDER BY query ASC"
    cursor.execute(sql)
    result = cursor.fetchall()
    return render_template('test_doc2vec.html', result=result)

@app.route('/test_nocluster')
def test_nocluster():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM testing WHERE typeir='nocluster' ORDER BY query ASC"
    cursor.execute(sql)
    result = cursor.fetchall()
    return render_template('test_nocluster.html', result=result)