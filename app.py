import streamlit as st
import pickle
import re
import nltk
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))
df = pd.read_csv("UpdatedResumeDataSet.csv")

def clean_resume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def convert_pdf_to_txt_pages(path):
    texts = []
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    size = 0
    c = 0
    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
      interpreter.process_page(page)
      t = retstr.getvalue()
      if c == 0:
        texts.append(t)
      else:
        texts.append(t[size:])
      c = c+1
      size = len(t)
    # text = retstr.getvalue()

    # fp.close()
    device.close()
    retstr.close()
    return texts, nbPages

# web app
def main():
    st.title("Resume Screening App")
    fig = plt.figure(figsize=(20,10))
    sns.countplot(x=df['Category'], hue=df['Category'], palette="husl")
    plt.xticks(rotation=90)
    st.pyplot(fig)

    pull_values = np.zeros(df['Category'].nunique())
    pull_values[0] = 0.1
    counts = df['Category'].value_counts()
    labels = counts.index

    fig2 = go.Figure(data=[go.Pie(labels=labels, values=counts, pull=pull_values)])
    fig2.update_layout(title_text="Unique Category Distribution")
    fig2.update_traces(hole=0.1)
    st.plotly_chart(fig2)

    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "pdf":
            text_data, nbPages = convert_pdf_to_txt_pages(uploaded_file)
            resume_txt = ""
            for i in range(nbPages):
               resume_txt = resume_txt + text_data[i]
        else:
           resume_txt = uploaded_file.read().decode()

        cleaned_resume = clean_resume(resume_txt)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)



# python main
if __name__ == "__main__":
    main()