import os
import PyPDF2
import re
from unidecode import unidecode

def obter_conteudo_dos_diarios_oficiais(path: str) -> str:
    # Leitura dos dados
    print("#Leitura: Lendo os diários oficiais em PDF\n")

    # Incializa o conteúdo de treino como string vazia
    conteudo_treino = ""
    # Para cada arquivo dentro de ./dos_treino
    for do in os.listdir(path):
        # Extrai o arquivo pdf
        pdf_file = open(f"{path}/{do}", "rb")
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
        conteudo_treino += unidecode("".join(re.split(r"O\s*DIÁRIO\s*OFICIAL\s*DOS\s*MUNICÍPIOS\s*DO\s*ESTADO\s*DE\s*PERNAMBUCO\s*É\s*UMA\s*SOLUÇÃO\s*VOLTADA\s*À\s*MODERNIZAÇÃO\s*E\s*TRANSPARÊNCIA\s*DA\s*GESTÃO\s*MUNICIPAL.\s*", pdf_content)[1:]))
        print("\n")
    return conteudo_treino