from flask import Flask,render_template,request
import matplotlib
import matplotlib.pyplot as plt
font={'size':22}
matplotlib.use('agg')
matplotlib.rc('font',**font)

import pickle
import joblib

cv=pickle.load(open('model_final.pkl','rb'))

app=Flask(__name__)

@app.route('/predict' , methods=("GET","POST"))
def predict():
  try:
    import requests
    from bs4 import BeautifulSoup
    HEADERS = ({'User-Agent':'Chrome/90.0.4430.212 Safari/537.36','Accept-Language': 'en-US, en;q=0.5'})
    def getdata(url):
      r = requests.get(url,headers=HEADERS,timeout=90)
      return r.text
    def html_code(url):
      htmldata = getdata(url)
      soup = BeautifulSoup(htmldata, 'html.parser')
      return (soup)
    product_name=request.form['product_name']
    url = "https://www.amazon.in/s?k=" + product_name
    soup = html_code(url)
    product=soup.find('a',{'class':'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal'})
    target_product='https://amazon.in'+product.get('href')
    soup=html_code(target_product)
    product_image=soup.find('div',{'class':'imgTagWrapper'}).find('img').get('src')
    product_title=soup.find('span',{'class':'a-size-large product-title-word-break'}).text
    product_price='₹'+soup.find('span',{'class':'a-price-whole'}).text.split('.')[0]
    product_description=soup.find('ul',{'class':'a-unordered-list a-vertical a-spacing-mini'}).find_all('li')
    product_description_list=[]
    for item in product_description:
      product_description_list.append(item.text)
    all_review_link='https://amazon.in'+soup.find('a',{'data-hook':'see-all-reviews-link-foot'}).get('href')
    soup=html_code(all_review_link)
    reviews_link='https://amazon.in'+soup.find('li',{'class':'a-last'}).find('a').get('href').split('&')[0]

    reviewlist=[]

    def get_reviews(soup):
      reviews = soup.find_all('div', {'data-hook': 'review'})
      try:
          for item in reviews:
              review =item.find('span', {'data-hook': 'review-body'}).text.strip()
              reviewlist.append(review)
      except:
          pass
      
    for x in range(1,20):
      soup = html_code(reviews_link+f"&pageNumber={x}")
      get_reviews(soup)
      if not soup.find('li', {'class': 'a-disabled a-last'}):
          pass
      else:
          break

    from wordcloud import WordCloud
    consolidated = " ".join(review for review in reviewlist)
    wordcloud = WordCloud(width=680,height=335,random_state=21,max_font_size=45,max_words=90).generate(consolidated)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    wordcloud.to_file("static/wordcloud.png")

    X_fresh = cv.transform(reviewlist).toarray()
    classifier = joblib.load('Classifier__Model')
    y_pred = classifier.predict(X_fresh).tolist()
    pos_count=0
    neg_count=0
    for i in y_pred:
      if(i==1):
        pos_count+=1
      else:
        neg_count+=1
    plt.clf()
    plt.bar(['Positive','Negative'],[pos_count,neg_count])
    plt.title('Review Analysis')
    plt.savefig('static/plot.png')

    return render_template('after.html',url_plot='/static/plot.png',wordcloud_plt='/static/wordcloud.png',name=product_title,url_image=product_image,price=product_price,description=product_description_list,nav_link=target_product)
  except:
     return render_template('busy.html')

@app.route('/')
def main():
    return render_template('index.html')

app.run(debug=True)