import sys # Biblioteca nativa para obter argumentos do terminal
from sklearn.feature_extraction.text import TfidfVectorizer # Biblioteca para vetorização de textos (NLP)
from sklearn.cluster import KMeans # Biblioteca para agrupamentos dos textos vetorizados
import PyPDF2 # Biblioteca para leitura dos DOs em PDF
import re # Biblioteca nativa para utilização de expressões regulares em Python
import numpy as np # Biblioteca para manipulação de vetores
import os # Biblioteca nativa para listagem de arquivos e diretórios
from sklearn.ensemble import RandomForestClassifier # Biblioteca para classificação das publicações
import spacy # Biblioteca para vetorização de textos, com base na semântica (NLP)
import pickle # Biblioteca nativa para salvar o treinamento

# Inicializa a random_forest como None
random_forest = None

# Define o modo de execução default como 'build'
mode = 'start' # Pode ser build ou start

# Define o número de clusters default como 4
num_clusters = 5

# Verifica se algum argumento foi passado ao executar o programa. Ex: python main.py build
if len(sys.argv) > 1:
    # build ou start
    mode = sys.argv[1]

print(f"Modo: {mode}, Número de clusters: {num_clusters}")

if mode == 'build':
    print('Modo build selecionado. Iniciando um novo treinamento\n')

else:
    try:
        # Caso haja um arquivo de treino salvo. Use-o na random_Forest
        with open('random_forest_training.pkl', 'rb') as file:
            random_forest = pickle.load(file)
            print("Carregando treinamento salvo\n")
    except:
        print("Nenhum treinamento encontrado. Inciando um novo treinamento\n")

# Define que o spacy deverá utilizar para analisar a semântica, como Português
nlp = spacy.load("pt_core_news_sm")

# Caso haja não haja treinamento salvo, inicie o treinamento
if random_forest is None:
##############################################################################################################################
    # Leitura dos dados
    print("#Leitura: Lendo os diários oficiais em PDF\n")

    # Incializa o conteúdo de treino como string vazia
    conteudo_treino = ""
    # Para cada arquivo dentro de ./dos_treino
    for do in os.listdir('./dos_treino'):
        # Extrai o arquivo pdf
        pdf_file = open(f"./dos_treino/{do}", "rb")
        # Lê o arquivo pdf
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Inicializa o conteúdo do arquivo com string vazia
        pdf_content = ""
        print(f"lendo {do}")

        # Itera as páginas do PDF
        for page_num in range(len(pdf_reader.pages)):
            print(f"Página: {page_num + 1}/{len(pdf_reader.pages)}", end="\r")

            # Extrai a página da iteração
            page = pdf_reader.pages[page_num]

            # Concatena todo o texto da página, em CAPS LOCK, no conteúdo total do PDF
            pdf_content += page.extract_text().upper()
        # Remove o cabeçalho do Diário Oficial lido e adiciona o resto do texto ao counteúdo de todos os diários
        conteudo_treino += ("".join(re.split(r"O\s*DIÁRIO\s*OFICIAL\s*DOS\s*MUNICÍPIOS\s*DO\s*ESTADO\s*DE\s*PERNAMBUCO\s*É\s*UMA\s*SOLUÇÃO\s*VOLTADA\s*À\s*MODERNIZAÇÃO\s*E\s*TRANSPARÊNCIA\s*DA\s*GESTÃO\s*MUNICIPAL.\s*", pdf_content)[1:]))
        print("\n")
        break
##############################################################################################################################
    # Pré-processamento dos dados
    print("#Pré-processamento: Extraindo lista de publicações e códigos identificadores\n")

    # A partir do conteúdo de todos os DOs, gera uma lista com as publicações, localizadas pelo CODIGO IDENTIFICADOR
    publicacoes_treino = re.split(r"CÓDIGO\s*IDENTIFICADOR:\s*[A-Za-z0-9]+", conteudo_treino)[:-1]

    # Também gera uma lista de mesmo tamanho contendo os códigos de cada publicação na respectiva ordem
    codigos_treino = list(map(lambda x:  " ".join(re.split(r"\s+", x)) ,re.findall(r"CÓDIGO\s*IDENTIFICADOR:\s*[A-Z0-9]+", conteudo_treino)))

    print('Vetorizando publicações e gerando clusters')
##############################################################################################################################
    # Processamento de linguagem natural
    print("#Processamento: Processando publicações (NLP)\n")

    # Vetoriza todas as publicações utilizando um algoritmo NLP, o Spacy, que analisa a semântica
    X = []
    for i, publicacao in enumerate(publicacoes_treino):
        print(f"Publicação: {i + 1}/{len(publicacoes_treino)}", end="\r")
        embedding = np.mean([token.vector for token in nlp(publicacao)], axis=0)
        X.append(embedding)

##############################################################################################################################
    # Agrupamento dos dados processados
    print("#Agrupamento: Agrupando as publicações processadas\n")

    # Agrupa a matriz gerada acima com base nos seu valores
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # Uma lista de todos os labels gerados para cada publicação
    labels = kmeans.labels_
    # Para cada label
    for i, label in enumerate(labels):
        # Imprima o índice da publicação, seu código e o cluster atribuído
        print(f"Publicacao: {i}, {codigos_treino[i]} - Cluster: {label}")

###############################################################################################################################
    # Extração de informações dos agrupamentos gerados
    print("#Visualização: Extraindo textos relevantes dos agrupamentos\n")

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
    vectorizer = TfidfVectorizer()
    cluster_keywords = {}
    for cluster_id, indices in cluster_indices.items():
        cluster_documents = [publicacoes_treino[i] for i in indices]
        cluster_vectorizer = TfidfVectorizer(stop_words='english')
        cluster_X = cluster_vectorizer.fit_transform(cluster_documents)
        cluster_nomes = cluster_vectorizer.get_feature_names_out()
        filtered_indices = (np.vectorize(len)(cluster_nomes) >= 5) & (~np.isin(cluster_nomes, palavras_nao_chave))
        filtered_keywords = np.array(cluster_nomes)[filtered_indices]
        cluster_termos_frequentes = np.asarray(cluster_X.sum(axis=0)).ravel()
        indices_ordenados = cluster_termos_frequentes.argsort()[::-1]
        cluster_top_keywords = [cluster_nomes[i] for i in indices_ordenados if cluster_nomes[i] in filtered_keywords][:4]
        cluster_keywords[cluster_id] = cluster_top_keywords

    # Imprime as palavras-chave para cada cluster
    for cluster_id, keywords in cluster_keywords.items():
        print(f"Cluster {cluster_id}, Publicacoes: {len(list(filter(lambda x: x == cluster_id, labels)))} / {len(labels)}, Palavras-chave: {', '.join(keywords)}")

###############################################################################################################################
    # Treinamento do classificador (random_forest)
    print("#Treinamento: Treinando classificador\n")

    # Define o Y, as saídas, como os rótulos gerados pelo agrupador. 
    # y_treino = list(map(lambda x: [1 if x == i else 0 for i in [n for n in range(num_clusters)]], labels))
    y_treino = labels

    X_treino = X
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    random_forest.fit(X_treino, y_treino)

    # Salva o treinamento da random forest em um arquivo .pkl, para que possa ser utilizado novamente, sem a necessidade de se treinar de novo
    with open('random_forest_training.pkl', 'wb') as file:
        pickle.dump(random_forest, file)

###############################################################################################################################
# Leitura dos dados de teste
print("#Leitura: Lendo diários a serem analisados em PDF\n")

conteudo_teste = ""
for do in os.listdir('./dos_teste'):
    pdf_file = open(f"./dos_teste/{do}", "rb")

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    pdf_content = ""
    print(f"lendo {do}")
    # Itera as páginas do PDF
    for page_num in range(len(pdf_reader.pages)):
        print(f"Página: {page_num + 1}/{len(pdf_reader.pages)}", end="\r")

        # Extrai a página da iteração
        page = pdf_reader.pages[page_num]

        # Concatena todo o texto da página, em CAPS LOCK, no conteúdo total do PDF
        pdf_content += page.extract_text().upper()
    conteudo_teste += ("".join(re.split(r"O\s*DIÁRIO\s*OFICIAL\s*DOS\s*MUNICÍPIOS\s*DO\s*ESTADO\s*DE\s*PERNAMBUCO\s*É\s*UMA\s*SOLUÇÃO\s*VOLTADA\s*À\s*MODERNIZAÇÃO\s*E\s*TRANSPARÊNCIA\s*DA\s*GESTÃO\s*MUNICIPAL.\s*", pdf_content)[1:]))
    print("\n")
    

##########################################################################################################################
# Pré-processamento dos dados de teste
print("#Pré-processamento: Extraindo lista de publicações e códigos identificadores dos dados de teste\n")

# A partir do conteúdo, gera uma lista com as publicações, localizadas pelo CODIGO IDENTIFICADOR
publicacoes_teste = re.split(r"CÓDIGO\s*IDENTIFICADOR:\s*[A-Za-z0-9]+", conteudo_teste)[:-1]

codigos_teste = list(map(lambda x:  " ".join(re.split(r"\s+", x)) ,re.findall(r"CÓDIGO\s*IDENTIFICADOR:\s*[A-Z0-9]+", conteudo_teste)))

###########################################################################################################################
# Processamento dos dados de teste
print("#Processamento: Vetorizando publicações de teste\n")
novas_publicacoes_vetorizadas = []
for i, publicacao in enumerate(publicacoes_teste):
    print(f"Publicação: {i + 1}/{len(publicacoes_teste)}", end="\r")
    embedding = np.mean([token.vector for token in nlp(publicacao)], axis=0)
    novas_publicacoes_vetorizadas.append(embedding)
print("\n")

#############################################################################################################################
# Classificação dos dados de teste
print("#Classificação: Classificando dados de teste processados\n")
# Obtem um array de arrys, contendo a probabilidade de cada publicação pertencer a cada classe
resultados = random_forest.predict_proba(novas_publicacoes_vetorizadas)

#############################################################################################################################
# Visualização dos resultados
print("#Resultados: Exibindo resultados\n")

# Define a porcentagme mínima para que uma publicação seja considerada de uma classe
threshold = 0.4
for i, v in enumerate(resultados):
    print(f"{codigos_teste[i]}, Clusters: {list(filter(lambda c: resultados[i][c] > threshold, [c for c in range(num_clusters)]))}, Predições: {resultados[i]}")