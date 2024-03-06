import os
import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FolderMonitor(FileSystemEventHandler):
    def __init__(self, folder_path, docs):
        super().__init__()
        self.folder_path = folder_path
        self.docs = docs

    def on_modified(self, event):
        if event.is_directory:
            return
        self.update_model()

    


    def update_model(self):
        print('Updating model...')
        model, tagged_data = train_doc2vec_model(self.docs)
        doc_vectors = [model.infer_vector(doc.words) for doc in tagged_data]
        average_vectors = [[np.mean(vector)] for vector in doc_vectors]
        average_vectors_array = np.array(average_vectors)
        cluster_labels = cluster_documents(average_vectors_array, 3)
        visualize_clusters(average_vectors_array, cluster_labels, 3)

  

   

def train_doc2vec_model(array_of_arrays):
    print('Training Doc2Vec model...')
    tagged_data = [TaggedDocument(words=d, tags=[str(i)]) for i, d in enumerate(array_of_arrays)]
    model = Doc2Vec(tagged_data, vector_size=100, window=2, min_count=1, epochs=100)
    return model, tagged_data

def cluster_documents(average_vectors_array, k=10):
    print('Clustering...')
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(average_vectors_array)
    return kmeans.labels_

def visualize_clusters(doc_vectors_array, cluster_labels, k):
    print('Visualizing clusters...')
    plt.figure(figsize=(8, 6))
    for i in range(k):
        indices = [index for index, label in enumerate(cluster_labels) if label == i]
        cluster_points = doc_vectors_array[indices]
        plt.scatter(indices, cluster_points, label=f'Cluster {i+1}')
    plt.title('t-SNE Visualization of Document Clusters (K-means)')
    plt.xlabel('Document Index')
    plt.ylabel('t-SNE Dimension')
    plt.legend()
    plt.show()

def main():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    folder_path = os.path.join(parent_directory, 'GP_Crawler', 'indexes')

    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                text = data.get('tokens', '')  # Assuming 'tokens' key contains the text
                docs.append(text)

    folder_monitor = FolderMonitor(folder_path, docs)
    folder_monitor.update_model()

    observer = Observer()
    observer.schedule(folder_monitor, folder_path, recursive=False)
    observer.start()
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
