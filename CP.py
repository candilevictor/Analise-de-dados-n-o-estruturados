import spacy
from gensim import corpora
from gensim.models import LdaModel

# Carregar o modelo para o idioma português
nlp = spacy.load('pt_core_news_sm')

# Caminho para o arquivo de texto
caminho_arquivo = "./noticia.txt"

# Abrir o arquivo e ler o conteúdo
with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
    texto = arquivo.read()
    
# Processar o texto
doc = nlp(texto)

# Função para análise sintática e extração de entidades nomeadas
def analisar(doc):
    for sentenca in doc.sents:
        print("Sentença:", sentenca.text)
        print("Análise sintática:")
        for token in sentenca:
            print(f"Token: {token.text}, Part-of-Speech: {token.pos_}, Dependência: {token.dep_}")
        print("Entidades nomeadas:")
        for entidade in sentenca.ents:
            print(f"Entidade: {entidade.text}, Tipo: {entidade.label_}")
        print()

# Exibir tokens
print("Tokens:")
for token in doc:
    print(token.text)

# Exibir as entidades nomeadas encontradas
print("\nEntidades Nomeadas:")
for entidade in doc.ents:
    print(f"Entidade: {entidade.text}, Tipo: {entidade.label_}")

# Analisar sintaxe e entidades nomeadas
print("\nAnálise Sintática e Entidades Nomeadas:")
analisar(doc)

# Tokenizar o texto
tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]

# Criar um dicionário com os tokens
dicionario = corpora.Dictionary([tokens])

# Criar um corpus
corpus = [dicionario.doc2bow(tokens)]

# Treinar o modelo LDA
lda_model = LdaModel(corpus, num_topics=3, id2word=dicionario)

# Exibir os tópicos
print("\nExtração de Tópicos:")
print("Tópicos mais relevantes:")
for idx, topic in lda_model.print_topics(-1):
    print(f'Tópico {idx}:')
    print(f'Palavras-chave: {", ".join(word.split("*")[1].strip() for word in topic.split("+"))}')
    print()

# Resumo Automático
print("\nResumo Automático:")

# Exibir as sentenças com sua pontuação (tamanho)
sentencas = [sent.text for sent in doc.sents]
tamanhos = [len(sent) for sent in sentencas]

# Ordenar as sentenças por pontuação (tamanho)
sentencas_ordenadas = [sentencas[i] for i in sorted(range(len(tamanhos)), key=lambda k: tamanhos[k], reverse=True)]

# Definir o número de sentenças no resumo
numero_sentencas_resumo = 3

# Criar o resumo
resumo = ' '.join(sentencas_ordenadas[:numero_sentencas_resumo])

# Exibir o resumo
print(resumo)
