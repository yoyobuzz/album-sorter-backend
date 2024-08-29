import requests
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from typing import List, Optional
from app.schemas import Face, Cluster, FaceCreate, ClusterCreate, PyObjectId
from pydantic import BaseModel, Field
from bson import ObjectId

def cosine_similarity(X, Y=None):
    if Y is None:
        Y = X
    X = np.asarray(X)
    Y = np.asarray(Y)
    dot_product = np.dot(X, Y.T)
    X_norm = np.linalg.norm(X, axis=1)[:, np.newaxis]
    Y_norm = np.linalg.norm(Y, axis=1)[np.newaxis, :]
    similarity = dot_product / (X_norm * Y_norm)
    return similarity

# Initialize the MTCNN face detector and Inception Resnet V1 model
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Function to extract face encodings from an image using MTCNN and Facenet
def extract_face_encodings(image, url: str) -> List[FaceCreate]:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(image_rgb)

    if boxes is None:
        print("No faces detected")
        return []

    face_encodings = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

        if x2 > x1 and y2 > y1:
            face = image_rgb[y1:y2, x1:x2]
            face_resized = cv2.resize(face, (160, 160))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float()
            face_tensor = (face_tensor - 127.5) / 128.0

            with torch.no_grad():
                face_encoding = model(face_tensor).numpy().flatten().tolist()
            face_data = FaceCreate(url=url, embedding=face_encoding)
            face_encodings.append(face_data)
    return face_encodings

# Function to process images from URLs
def process_images_from_urls(image_urls: List[str]) -> List[FaceCreate]:
    faces = []

    for url in image_urls:
        try:
            response = requests.get(url)
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            face_encodings = extract_face_encodings(image, url=url)
            for face in face_encodings:
                face.url = url  # Assign the URL to each face
            faces.extend(face_encodings)
        except Exception as e:
            print(f"Error processing image {url}: {e}")
    return faces

# Function to cluster faces based on similarity
def cluster_faces(faces: List[FaceCreate], similarity_threshold=0.6) -> List[Cluster]:
    clusters = []

    for face in faces:
        encoding = np.array(face.embedding)
        max_similarity = -1
        best_cluster_index = None

        for j, cluster in enumerate(clusters):
            cluster_center = np.array(cluster.cluster_centre)
            similarity = cosine_similarity([encoding], [cluster_center])[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                best_cluster_index = j

        if max_similarity > similarity_threshold and best_cluster_index is not None:
            best_cluster = clusters[best_cluster_index]
            best_cluster.face_images.append(Face(embedding=face.embedding, url=face.url))
            n = len(best_cluster.face_images)
            updated_center = (encoding + cluster_center * (n - 1)) / n
            best_cluster.cluster_centre = updated_center.tolist()
        else:
            new_cluster = Cluster(
                id=str(ObjectId()),
                cluster_centre=face.embedding,
                face_images=[Face(id=str(ObjectId()),embedding=face.embedding, url=face.url)]
            )
            clusters.append(new_cluster)
    return clusters

# Main execution function
def process_urls(image_urls: List[str]) -> List[Cluster]:
    faces = process_images_from_urls(image_urls)
    clusters = cluster_faces(faces)
    return clusters

# import requests
# import cv2
# import numpy as np
# from facenet_pytorch import InceptionResnetV1, MTCNN
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import List
# from app.schemas import *

# # Initialize the MTCNN face detector and Inception Resnet V1 model
# mtcnn = MTCNN(keep_all=True)
# model = InceptionResnetV1(pretrained='vggface2').eval()

# # Function to extract face encodings from an image using MTCNN and Facenet
# def extract_face_encodings(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     boxes, _ = mtcnn.detect(image_rgb)

#     if boxes is None:
#         print("No faces detected")
#         return [], []

#     face_encodings = []
#     faces = []
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box)
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

#         if x2 > x1 and y2 > y1:
#             face = image_rgb[y1:y2, x1:x2]
#             face_resized = cv2.resize(face, (160, 160))
#             face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float()
#             face_tensor = (face_tensor - 127.5) / 128.0

#             with torch.no_grad():
#                 face_encoding = model(face_tensor).numpy().flatten()
#             faces.append(face)
#             face_encodings.append(face_encoding)
#     return face_encodings, faces

# # Function to process images from URLs
# def process_images_from_urls(image_urls: List[str]):
#     encodings = []
#     image_paths = []
#     faces = []

#     for url in image_urls:
#         try:
#             response = requests.get(url)
#             image = np.asarray(bytearray(response.content), dtype="uint8")
#             image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#             face_encodings, image_faces = extract_face_encodings(image)
#             if face_encodings:
#                 encodings.extend(face_encodings)
#                 image_paths.extend([url] * len(face_encodings))
#                 faces.extend(image_faces)
#         except Exception as e:
#             print(f"Error processing image {url}: {e}")
#     return encodings, image_paths, faces

# # Function to cluster faces based on similarity
# def cluster_faces(encodings, image_paths: List[str], faces, similarity_threshold=0.6):
#     clusters = []
    
#     for i, encoding in enumerate(encodings):
#         max_similarity = -1
#         best_cluster_index = None
        
#         for j, cluster in enumerate(clusters):
#             cluster_center = cluster['cluster_center']
#             similarity = cosine_similarity([encoding], [cluster_center])[0][0]
            
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 best_cluster_index = j
        
#         if max_similarity > similarity_threshold and best_cluster_index is not None:
#             best_cluster = clusters[best_cluster_index]
#             best_cluster['face_images'].append({
#                 'embedding': encoding,
#                 'url': image_paths[i]
#             })
#             n = len(best_cluster['face_images'])
#             best_cluster['cluster_center'] = (encoding + best_cluster['cluster_center'] * (n - 1)) / n
#         else:
#             clusters.append({
#                 'cluster_center': encoding,
#                 'face_images': [{
#                     'embedding': encoding,
#                     'url': image_paths[i]
#                 }]
#             })
#     return clusters

# # Function to print cluster details
# def print_clusters(clusters):
#     for i, cluster in enumerate(clusters):
#         print(f"Cluster {i + 1}:")
#         print(f"  Center: {cluster['cluster_center']}")
#         print("  Face Images:")
#         for face_image in cluster['face_images']:
#             print(f"    - URL: {face_image['url']}")
#             print(f"      Embeddings: {face_image['embedding']}")
#         print("\n")

# # Main execution function
# def process_urls(image_urls: List[str]):
#     encodings, image_paths, faces = process_images_from_urls(image_urls)
#     clusters = cluster_faces(encodings, image_paths, faces)
#     return clusters
#     # print_clusters(clusters)
