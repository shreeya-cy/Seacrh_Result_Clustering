

from elasticsearch import Elasticsearch
import os
import pandas as pd
import nltk
from sklearn.decomposition import PCA
nltk.download('punkt')
nltk.download('stopwords')
import matplotlib.pyplot  as plt
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def index_search(query):
    es = Elasticsearch("https://localhost:9200",verify_certs=False, http_auth=('elastic', 'X3-ad2pd_g5bjLYxvfJW'))
    print(es.ping())
    if not es.indices.exists(index='practice_index'):
        es.indices.create(index='practice_index', ignore=400)
        folder_path = '/Users/shreeyacy/Desktop/IR_P05/data'
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt') and os.path.isfile(os.path.join(folder_path, f))]
        for txt_file in files:
            file_path = os.path.join(folder_path, txt_file)
            with open(file_path, 'r') as file:
                content = file.read()
                document = {
                    'filename': txt_file,
                    'content': content
                }
                es.index(index='practice_index', body=document)

    search_query = {
            'query': {
                'match': {
                    'content': query
                }
            },
            'size': 20
        }
    results = es.search(index='practice_index', body=search_query)
    if(results['hits']['total']['value'] == 0):
        print("No results found")

    # Extract and return relevant information from the search results
    relevant_documents = []
    for hit in results['hits']['hits']:
        document_info = {
            'filename': hit['_source']['filename'],
            'content': hit['_source']['content']
        }
        relevant_documents.append(document_info)
    return relevant_documents

def get_clusters(k, relevant_documents):
    data = pd.DataFrame(relevant_documents)
    corpus = data['content'].tolist()
    original_corpus = corpus

    for doc in corpus:
        index = corpus.index(doc)
        corpus[index] = corpus[index].replace(u'\ufffd', ' ')
        corpus[index] = corpus[index].replace(',', '')
        corpus[index] = corpus[index].rstrip('\n')
        corpus[index] = corpus[index].casefold()

        listOfTokens = word_tokenize(corpus[index])
        filtered_tokens = [word for word in listOfTokens if word.lower() not in stopwords.words('english')]
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        corpus[index] = ' '.join(stemmed_tokens)


    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    tf_idf = pd.DataFrame(data = X.toarray(), columns=vectorizer.get_feature_names_out())

    final_df = tf_idf


    kmeans = KMeans(n_clusters=k,n_init=10, random_state=42)
    kmeans.fit(final_df)

    labels = kmeans.labels_
    documents = data['filename'].tolist()

    result_df = pd.DataFrame({'Document': documents, 'Cluster': labels})
    print_df = pd.DataFrame({'Document': documents, 'Cluster': labels, 'Content': original_corpus})
    print_df=print_df.reset_index(drop=True)
    print_df["doc_id"]=print_df.index

    #

    #printing the retrieved document titles with their cluster number
    print(print_df)


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    result_df['PCA1'] = X_pca[:, 0]
    result_df['PCA2'] = X_pca[:, 1]

    # Plot the scatterplot and add the document numbers as labels

    for cluster in range(k):
        cluster_data = result_df[result_df['Cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')
        for i, document in cluster_data.iterrows():
            plt.annotate(f"Doc {i}", (document['PCA1'], document['PCA2']))

    plt.title('Clusters of Documents')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('plot.png')
    plt.show()

    #convert print_df to json
    print_df = print_df.to_dict(orient="records")
    return print_df


if __name__ == '__main__':
    query = "india"
    k = 3
    relevant_documents = index_search(query)
    get_clusters(k, relevant_documents)
