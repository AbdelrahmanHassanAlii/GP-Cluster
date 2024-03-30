import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json

class FolderMonitor(FileSystemEventHandler):
    def __init__(self, folder_path, data_dir, output_directory):
        super().__init__()
        self.folder_path = folder_path
        self.data_dir = data_dir
        self.output_directory = output_directory
        self.docs = self.read_documents()

    def read_documents(self):
        docs = []
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    docs.append((data['tokens'], data['docname']))
        return docs

    def on_modified(self, event):
        if event.is_directory:
            return
        self.update_model()

    def update_model(self):
        print('Updating model...')
        model, tagged_data = self.train_doc2vec_model(self.docs)
        doc_vectors = np.array([model.infer_vector(doc[0]) for doc in tagged_data])
        cluster_labels = self.cluster_documents(doc_vectors, k=3)
        self.persist_cluster_labels(cluster_labels)

    def train_doc2vec_model(self, docs):
        print('Training Doc2Vec model...')
        tagged_data = [TaggedDocument(words=doc[0], tags=[str(i)]) for i, doc in enumerate(docs)]
        model = Doc2Vec(tagged_data, vector_size=100, window=2, min_count=1, epochs=100)
        return model, tagged_data

    def cluster_documents(self, doc_vectors, k=10):
        print('Clustering...')
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(doc_vectors)
        return kmeans.labels_


    def persist_cluster_labels(self, cluster_labels):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        # Create a folder for each cluster and put the documents in the folder
        for i, cluster_label in enumerate(cluster_labels):
            cluster_dir = os.path.join(self.output_directory, f"cluster_{cluster_label}")
            if not os.path.exists(cluster_dir):
                os.makedirs(cluster_dir)

            # Get the real document file name from the data directory
            document_file = next((doc[1] for doc in self.docs if str(i) in doc[1]), None)
            if document_file:
                # Fix file path format and append ".json"
                document_file = os.path.join(self.data_dir, document_file.replace("\\", "/") + ".json")
                shutil.copy(document_file, os.path.join(cluster_dir, os.path.basename(document_file)))

    # def persist_cluster_labels(self, cluster_labels):
    #     if not os.path.exists(self.output_directory):
    #         os.makedirs(self.output_directory)

    #     # Create a folder for each cluster and put the documents in the folder
    #     for i, cluster_label in enumerate(cluster_labels):
    #         cluster_dir = os.path.join(self.output_directory, f"cluster_{cluster_label}")
    #         if not os.path.exists(cluster_dir):
    #             os.makedirs(cluster_dir)

    #         # Get the real document file name from the data directory
    #         document_file = next((doc[1] for doc in self.docs if str(i) in doc[1]), None)
    #         if document_file:
    #             shutil.copy(os.path.join(self.data_dir, document_file), os.path.join(cluster_dir, document_file))

    # def persist_cluster_labels(self, cluster_labels):
    #     if not os.path.exists(self.output_directory):
    #         os.makedirs(self.output_directory)

    #     # Create a folder for each cluster and put the documents in the folder
    #     for i, cluster_label in enumerate(cluster_labels):
    #         cluster_dir = os.path.join(self.output_directory, f"cluster_{cluster_label}")
    #         if not os.path.exists(cluster_dir):
    #             os.makedirs(cluster_dir)

    #         # Get the real document file name from the data directory
    #         document_file = next((doc[1].split('/')[-1] for doc in self.docs if str(i) in doc[1]), None)
    #         if document_file:
    #             shutil.copy(os.path.join(self.data_dir, document_file), os.path.join(cluster_dir, document_file))


def visualize_clusters(doc_vectors, cluster_labels, k):
    print('Visualizing clusters...')
    tsne = TSNE(n_components=2, random_state=0)
    tsne_data = tsne.fit_transform(doc_vectors)

    plt.figure(figsize=(8, 6))
    for i in range(k):
        indices = np.where(cluster_labels == i)
        plt.scatter(tsne_data[indices, 0], tsne_data[indices, 1], label=f'Cluster {i+1}')
    plt.title('t-SNE Visualization of Document Clusters (K-means)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()

def main():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, os.pardir))  # Go up one level to project root
    folder_path = 'D:/Hasssan/My-Work(2023 + 2024)/GP/codes/Crawler/indexes'
    data_dir = 'D:/Hasssan/My-Work(2023 + 2024)/GP/codes/Crawler/data'
    output_directory = os.path.join(os.curdir, 'Clusters')

    # Raise an error if the folder does not exist
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder '{folder_path}' does not exist.")

    folder_monitor = FolderMonitor(folder_path, data_dir, output_directory)
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
