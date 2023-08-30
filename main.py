from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import PyPDF2
import re
from unidecode import unidecode

pdf_file = open("./diario-2023-08-29.pdf", "rb")

pdf_reader = PyPDF2.PdfReader(pdf_file)

text = ""

for page_num in range(len(pdf_reader.pages)):
    # if page_num >= 10: break
    print(f"Página: {page_num}")
    page = pdf_reader.pages[page_num]
    text += page.extract_text().upper()

text = unidecode("".join(re.split(r"O\s*DIÁRIO\s*OFICIAL\s*DOS\s*MUNICÍPIOS\s*DO\s*ESTADO\s*DE\s*PERNAMBUCO\s*É\s*UMA\s*SOLUÇÃO\s*VOLTADA\s*À\s*MODERNIZAÇÃO\s*E\s*TRANSPARÊNCIA\s*DA\s*GESTÃO\s*MUNICIPAL.\s*", text)[1:]))

publicacoes = re.split(r"CODIGO\s*IDENTIFICADOR:\s*[A-Z0-9]+", text)[:-1]

codigos = list(map(lambda x:  " ".join(re.split(r"\s+", x)) ,re.findall(r"CODIGO\s*IDENTIFICADOR:\s*[A-Z0-9]+", text)))

data = publicacoes

num_clusters = 4

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_

for i, label in enumerate(labels):
    print(f"Publicacao: {i}, {codigos[i]} - Cluster: {label}")