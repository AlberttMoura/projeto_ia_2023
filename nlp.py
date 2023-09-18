import numpy as np # Biblioteca para manipulação de vetores
import spacy # Biblioteca para vetorização de textos, com base na semântica (NLP)
import asyncio

def nlp_spacy(publicacoes: list[str]) -> list:
  # Define que o spacy deverá utilizar para analisar a semântica, como Português
  nlp = spacy.load("pt_core_news_sm")

  X = []
  for i, publicacao in enumerate(publicacoes):
    print(f"Publicação: {i + 1}/{len(publicacoes)}", end="\r")
    embedding = np.mean([token.vector for token in nlp(publicacao)], axis=0)
    X.append(embedding)
  return X

async def processar_publicacao(publicacao: str) -> list:
  print('oi')
  nlp = spacy.load('pt_core_news_sm')
  embeddding = np.mean([token.vector for token in nlp(publicacao)], axis=0)
  return embeddding

async def nlp_spacy_async(publicacoes: list[str]) -> list:
  X = []
  tasks = [processar_publicacao(publicacao=publicacao) for publicacao in publicacoes]
  embeddings = await asyncio.gather(*tasks)
  X.extend(embeddings)
  return X
