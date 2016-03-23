from bs4 import BeautifulSoup
import cluster
import time
from datetime import datetime

start_time = datetime.utcnow()
begin_time = time.localtime()
print("[INFO][%s]Begin"%(str(time.strftime("%X", begin_time))))
with open("./data/data.xml") as f:
    content = f.read()
    soup = BeautifulSoup(content, 'xml')

# get all articles node
articles = soup.find_all('PubmedArticle')
articles_num = len(articles)
print('[INFO]There are %d articles'%(articles_num))

titles = [None for i in range(articles_num)]
for i in xrange(articles_num):
    if articles[i].Title is not None:
        titles[i] = articles[i].ArticleTitle.get_text().encode('utf-8')

abstracts = [[] for i in range(articles_num)]
for i in xrange(articles_num):
    if articles[i].Abstract is not None:
        abstracts[i] = [text.get_text()
                     for text in articles[i].find_all('AbstractText')]
    abstracts[i] = "\n".join(abstracts[i])

info = [[i,titles[i],abstracts[i]] for i in range(articles_num)
        if abstracts[i] != ""]

print('[INFO]Data loaded')
# for i in info:
#     print(str(i)+'\n')

cluster.clust(info)
end_time = datetime.utcnow()
td = (end_time - start_time)
print("[INFO]In %s seconds"%(str(td.seconds)))
end_time = time.localtime()
print('[INFO][%s]Done!:)'%(str(time.strftime("%X", end_time))))
