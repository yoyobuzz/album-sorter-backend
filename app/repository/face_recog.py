import requests
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from typing import List, Optional
from app.schemas import Face, Cluster, FaceCreate, ClusterCreate
from bson import ObjectId

# Initialize the MTCNN face detector and Inception Resnet V1 model
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

def cosine_similarity(X, Y=None):
    """
    Compute cosine similarity between vectors X and Y.
    """
    if Y is None:
        Y = X
    X = np.asarray(X)
    Y = np.asarray(Y)
    dot_product = np.dot(X, Y.T)
    X_norm = np.linalg.norm(X, axis=1)[:, np.newaxis]
    Y_norm = np.linalg.norm(Y, axis=1)[np.newaxis, :]
    return dot_product / (X_norm * Y_norm)

def extract_face_encodings_with_url(image, url: str) -> List[FaceCreate]:
    """
    Extract face encodings from an image using MTCNN and Facenet.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(image_rgb)
    if boxes is None:
        return []

    face_encodings = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = image_rgb[max(0, y1):min(y2, image_rgb.shape[0]), max(0, x1):min(x2, image_rgb.shape[1])]
        face_resized = cv2.resize(face, (160, 160))
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float()
        face_tensor = (face_tensor - 127.5) / 128.0

        with torch.no_grad():
            face_encoding = model(face_tensor).numpy().flatten().tolist()
        face_encodings.append(FaceCreate(url=url, embedding=face_encoding))
    
    return face_encodings

def process_images_from_urls(image_urls: List[str]) -> List[FaceCreate]:
    """
    Process a list of image URLs and extract face encodings.
    """
    faces = []
    for url in image_urls:
        try:
            response = requests.get(url)
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            faces.extend(extract_face_encodings_with_url(image, url=url))
        except Exception as e:
            print(f"Error processing image {url}: {e}")
    return faces

def cluster_faces(faces: List[FaceCreate], clusters: List[Cluster], similarity_threshold: float = 0.6) -> List[Cluster]:
    """
    Cluster faces based on similarity.
    """
    modified_cluster_id = set()
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
            best_cluster.face_images.append(Face(id=str(ObjectId()), embedding=face.embedding, url=face.url))
            n = len(best_cluster.face_images)
            updated_center = (encoding + np.array(best_cluster.cluster_centre) * (n - 1)) / n
            best_cluster.cluster_centre = updated_center.tolist()
            modified_cluster_id.add(best_cluster.id)
        else:
            new_cluster = Cluster(
                id=str(ObjectId()),
                cluster_centre=face.embedding,
                face_images=[Face(id=str(ObjectId()), embedding=face.embedding, url=face.url)]
            )
            clusters.append(new_cluster)
            modified_cluster_id.add(new_cluster.id)

    return [cluster for cluster in clusters if cluster.id in modified_cluster_id]

def process_urls(image_urls: List[str], clusters: List[Cluster]) -> List[Cluster]:
    """
    Process images from URLs and cluster them.
    """
    faces = process_images_from_urls(image_urls)
    return cluster_faces(faces, clusters)

def extract_face_encoding(image) -> Optional[Face]:
    """
    Extract a single face encoding from an image.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(image_rgb)
    if boxes is None:
        return None

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = image_rgb[max(0, y1):min(y2, image_rgb.shape[0]), max(0, x1):min(x2, image_rgb.shape[1])]
        face_resized = cv2.resize(face, (160, 160))
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float()
        face_tensor = (face_tensor - 127.5) / 128.0

        with torch.no_grad():
            face_encoding = model(face_tensor).numpy().flatten().tolist()

        return Face(embedding=face_encoding, url="")
    return None

def find_similar_clusters(input_encoding: List[float], clusters: List[Cluster], threshold: float = 0.5) -> List[Cluster]:
    """
    Find clusters similar to the given face encoding.
    """
    return [
        cluster for cluster in clusters
        if cosine_similarity([input_encoding], [np.array(cluster.cluster_centre)])[0][0] >= threshold
    ]

def search_photos(query_embedding: List[float], clusters: List[Cluster], similarity_threshold: float = 0.5) -> List[str]:
    """
    Search for photos in clusters similar to the query embedding.
    """
    similar_clusters = find_similar_clusters(query_embedding, clusters, similarity_threshold)
    similar_image_urls = set()

    for cluster in similar_clusters:
        for face_image in cluster.face_images:
            similarity = cosine_similarity([query_embedding], [np.array(face_image.embedding)])[0][0]
            if similarity > similarity_threshold:
                similar_image_urls.add(face_image.url)
    
    return list(similar_image_urls)

def capture_face_encoding(url: str) -> Optional[Face]:
    """
    Capture a face encoding from a given image URL.
    """
    response = requests.get(url)
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return extract_face_encoding(image)

def find_images(clusters: List[Cluster], query_image_url: str) -> List[str]:
    """
    Find images in clusters that match the face in the provided query image URL.
    """
    query_face = capture_face_encoding(query_image_url)
    if query_face:
        return search_photos(query_face.embedding, clusters)
    return []