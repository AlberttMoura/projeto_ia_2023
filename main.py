from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import PyPDF2
import re
from unidecode import unidecode
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

conteudo = ""
count = 0
for do in os.listdir('./dos'):
    if count == 1:
        break
    pdf_file = open(f"./dos/{do}", "rb")

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    pdf_content = ""
    print(f"lendo {do}")
    # Itera as páginas do PDF
    for page_num in range(len(pdf_reader.pages)):
        if page_num >= 30: break
        print(f"Página: {page_num}")

        # Extrai a página da iteração
        page = pdf_reader.pages[page_num]

        # Concatena todo o texto da página, em CAPS LOCK, no conteúdo total do PDF
        pdf_content += page.extract_text().upper()

    conteudo += unidecode("".join(re.split(r"O\s*DIÁRIO\s*OFICIAL\s*DOS\s*MUNICÍPIOS\s*DO\s*ESTADO\s*DE\s*PERNAMBUCO\s*É\s*UMA\s*SOLUÇÃO\s*VOLTADA\s*À\s*MODERNIZAÇÃO\s*E\s*TRANSPARÊNCIA\s*DA\s*GESTÃO\s*MUNICIPAL.\s*", pdf_content)[1:]))
    count +=1

# A partir do conteúdo, gera uma lista com as publicações, localizadas pelo CÓDIGO IDENTIFICADOR
publicacoes = re.split(r"CODIGO\s*IDENTIFICADOR:\s*[A-Za-z0-9]+", conteudo)[:-1]

# Também gera uma lista de mesmo tamanho contendo os códigos de cada publicação na respectiva ordem
codigos = list(map(lambda x:  " ".join(re.split(r"\s+", x)) ,re.findall(r"CODIGO\s*IDENTIFICADOR:\s*[A-Z0-9]+", conteudo)))

# Atribui a data, dados de leitura, a lista de publicações do diário
data = publicacoes

# Define um número de agrupamentos a serem formados. Não existe valor exato
num_clusters = 5

# Vetoriza a lista de textos. Fazendo uma relação de TODAS as palavras com TODOS os textos e dando uma uma "nota"(valor do vetor) para cada texto na dimensão de cada palavra
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)

# Agrupa a matriz gerada acima com base nos seu valores
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Uma lista de todos os labels gerados para cada publicação
labels = kmeans.labels_

# Para cada label
for i, label in enumerate(labels):
    # Imprima o índice da publicação, seu código e o cluster atribuído
    print(f"Publicacao: {i}, {codigos[i]} - Cluster: {label}")

# Obtém os índices dos documentos em cada cluster
cluster_indices = {}
for cluster_id in range(num_clusters):
    cluster_indices[cluster_id] = [i for i, label in enumerate(labels) if label == cluster_id]

# Palavras para não incluir nas palavras chave
palavras_nao_chave = ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
                      "secretaria", "portaria", "municipio", "municipal", "estado", "estadual", "prefeito", "prefeitura", "governo", "vinte", "total", "pernambuco",
                      "qualquer", "todos", "todas", "tudo", "silva", "publicado", "escada", "pesqueira", "limoeiro", "salgueiro", "publico", "batista",
                      "barbosa", "michely", "marcela", "maria", "cedro", "paulista", "prefeita", "leite", "quental",
                      "cabo", "santo", "agostinho", "goiana", "secretario", "paudalho"
                      ]

# Obtém as palavras mais frequentes em cada cluster
cluster_keywords = {}
for cluster_id, indices in cluster_indices.items():
    cluster_documents = [publicacoes[i] for i in indices]
    cluster_vectorizer = TfidfVectorizer(stop_words='english')
    cluster_X = cluster_vectorizer.fit_transform(cluster_documents)
    cluster_nomes = cluster_vectorizer.get_feature_names_out()
    filtered_indices = (np.vectorize(len)(cluster_nomes) >= 5) & (~np.isin(cluster_nomes, palavras_nao_chave))
    filtered_keywords = np.array(cluster_nomes)[filtered_indices]
    cluster_termos_frequentes = np.asarray(cluster_X.sum(axis=0)).ravel()
    indices_ordenados = cluster_termos_frequentes.argsort()[::-1]
    cluster_top_keywords = [cluster_nomes[i] for i in indices_ordenados if cluster_nomes[i] in filtered_keywords][:7]
    cluster_keywords[cluster_id] = cluster_top_keywords

# Imprime as palavras-chave para cada cluster
for cluster_id, keywords in cluster_keywords.items():
    print(f"Cluster {cluster_id}, Publicacoes: {len(list(filter(lambda x: x == cluster_id, labels)))} / {len(labels)}, Palavras-chave: {', '.join(keywords)}")


# Classificador

X_treino = publicacoes
y_treino = labels
labels = list(map(lambda x: [1 if x == i else 0 for i in [n for n in range(num_clusters)]], labels))

# Tokeniza as entradas (transforma palavras em números)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(publicacoes)
sequences = tokenizer.texts_to_sequences(publicacoes)

# Tamanho do vocabulário, incluindo token Out Of Vocabulary (OOV)
vocab_size = len(tokenizer.word_index) + 1

# Faz com que todas as sequencias (inputs tokenizados), possuam o mesmo tamanho
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Codifica os labels
all_labels = [x for x in range(num_clusters)]

# Mapea cada label a um índice
label_encoder = {label: idx for idx, label in enumerate(all_labels)}

# Converte os labels para uma representação binário multi-label, para entradas associadas a mais e 1 label
# É como uma matriz de 0 para ausência do label e 1 para a presença

# Converte os dados de entrada e seus labels para uma matriz que possa ser utilazada pela Machine Learning
X = tf.constant(padded_sequences)

# Matriz com todas as entradas e seu mapeamento para os labels. Ex: [[0, 0, 1, 1, 0], ..., [1, 0, 0, 1, 0]]
y = tf.constant(labels)

# Define o modelo (rede neural)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=len(all_labels), input_length=max_length),
    tf.keras.layers.LSTM(1),
    tf.keras.layers.Dense(len(all_labels), activation='sigmoid')
])

# Compila o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treina o modelo
model.fit(X, y, epochs=1, batch_size=1)

# DO de teste

conteudo2 = ""
for do in os.listdir('./dos_teste'):
    pdf_file = open(f"./dos_teste/{do}", "rb")

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    pdf_content = ""
    print(f"lendo {do}")
    # Itera as páginas do PDF
    for page_num in range(len(pdf_reader.pages)):
        if page_num >= 20: break
        print(f"Página: {page_num}")

        # Extrai a página da iteração
        page = pdf_reader.pages[page_num]

        # Concatena todo o texto da página, em CAPS LOCK, no conteúdo total do PDF
        pdf_content += page.extract_text().upper()

    conteudo2 += unidecode("".join(re.split(r"O\s*DIÁRIO\s*OFICIAL\s*DOS\s*MUNICÍPIOS\s*DO\s*ESTADO\s*DE\s*PERNAMBUCO\s*É\s*UMA\s*SOLUÇÃO\s*VOLTADA\s*À\s*MODERNIZAÇÃO\s*E\s*TRANSPARÊNCIA\s*DA\s*GESTÃO\s*MUNICIPAL.\s*", pdf_content)[1:]))

# A partir do conteúdo, gera uma lista com as publicações, localizadas pelo CÓDIGO IDENTIFICADOR
publicacoes2 = re.split(r"CODIGO\s*IDENTIFICADOR:\s*[A-Za-z0-9]+", conteudo)[:-1]

# Também gera uma lista de mesmo tamanho contendo os códigos de cada publicação na respectiva ordem
codigos2 = list(map(lambda x:  " ".join(re.split(r"\s+", x)) ,re.findall(r"CODIGO\s*IDENTIFICADOR:\s*[A-Z0-9]+", conteudo)))


# Gerar predições
new_texts = publicacoes
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length)
predictions = model.predict(tf.constant(new_padded_sequences))
print(predictions)

# Mapeia as predições com os labels
threshold = 0.48  # Limiar de probabilidade da string pertencer ao tema
predicted_labels = [[all_labels[i] for i in range(len(all_labels)) if prediction[i] > threshold] for prediction in predictions]

def decode_labels(label_indices):
    return [idx for idx in label_indices]

print("Publicações e seus clusters correspondentes:")
for i in range(len(new_texts)):
    print(f'Publicação: "{codigos[i]}", Cluster: {decode_labels(predicted_labels[i])}')