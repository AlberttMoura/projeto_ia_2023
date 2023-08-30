from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import PyPDF2
import re
from unidecode import unidecode

# Instancia o arquivo PDF do Diário Oficial a ser lido. Obs: Trocar nome do arquivo para um que exista no diretório raiz
pdf_file = open("./diario-2023-08-29.pdf", "rb")

pdf_reader = PyPDF2.PdfReader(pdf_file)

# Todo o conteúdo do PDF em forma de string
conteudo = ""

# Itera as páginas do PDF
for page_num in range(len(pdf_reader.pages)):
    # if page_num >= 10: break
    print(f"Página: {page_num}")

    # Extrai a página da iteração
    page = pdf_reader.pages[page_num]

    # Concatena todo o texto da página, em CAPS LOCK, no conteúdo total do PDF
    conteudo += page.extract_text().upper()

# Remove o cabeçalho inicial do Diário oficial
conteudo = unidecode("".join(re.split(r"O\s*DIÁRIO\s*OFICIAL\s*DOS\s*MUNICÍPIOS\s*DO\s*ESTADO\s*DE\s*PERNAMBUCO\s*É\s*UMA\s*SOLUÇÃO\s*VOLTADA\s*À\s*MODERNIZAÇÃO\s*E\s*TRANSPARÊNCIA\s*DA\s*GESTÃO\s*MUNICIPAL.\s*", conteudo)[1:]))

# A partir do conteúdo, gera uma lista com as publicações, localizadas pelo CÓDIGO IDENTIFICADOR
publicacoes = re.split(r"CODIGO\s*IDENTIFICADOR:\s*[A-Z0-9]+", conteudo)[:-1]

# Também gera uma lista de mesmo tamanho contendo os códigos de cada publicação na respectiva ordem
codigos = list(map(lambda x:  " ".join(re.split(r"\s+", x)) ,re.findall(r"CODIGO\s*IDENTIFICADOR:\s*[A-Z0-9]+", conteudo)))

# Atribui a data, dados de leitura, a lista de publicações do diário
data = publicacoes

# Define um número de agrupamentos a serem formados. Não existe valor exato
num_clusters = 4

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