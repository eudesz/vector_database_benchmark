import os
import requests
import h5py
import numpy as np
from tqdm import tqdm

class Dataset:
    def __init__(self, name="glove-100-angular"):
        self.name = name
        self.url = f"http://ann-benchmarks.com/{name}.hdf5"
        self.path = os.path.join("data", f"{name}.hdf5")
        
        self.train_vectors = None
        self.test_vectors = None
        self.neighbors = None
        self.distances = None
        self.dimension = 0

    def download(self):
        if not os.path.exists("data"):
            os.makedirs("data")
            
        if os.path.exists(self.path):
            print(f"Dataset {self.name} already exists.")
            return

        print(f"Downloading {self.name} from {self.url}...")
        response = requests.get(self.url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(self.path, 'wb') as f, tqdm(
            desc=self.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)

    def load(self):
        self.download()
        print(f"Loading {self.name}...")
        with h5py.File(self.path, 'r') as f:
            self.train_vectors = list(f['train'])
            self.test_vectors = list(f['test'])
            self.neighbors = list(f['neighbors'])
            self.distances = list(f['distances'])
            
        self.train_vectors = np.array(self.train_vectors)
        self.test_vectors = np.array(self.test_vectors)
        self.dimension = self.train_vectors.shape[1]
        
        # Normalize for Angular/Cosine if needed
        if 'angular' in self.name:
            print("Normalizing vectors for Angular distance...")
            self.train_vectors = self._normalize(self.train_vectors)
            self.test_vectors = self._normalize(self.test_vectors)

        print(f"Loaded {len(self.train_vectors)} train vectors, {len(self.test_vectors)} test vectors. Dim: {self.dimension}")

    def _normalize(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def get_ground_truth(self, k=10):
        """Returns the top K indices from ground truth"""
        return np.array(self.neighbors)[:, :k]

if __name__ == "__main__":
    ds = Dataset()
    ds.load()
