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
def extract_face_encodings_with_url(image, url: str) -> List[FaceCreate]:
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
            face_encodings = extract_face_encodings_with_url(image, url=url)
            for face in face_encodings:
                face.url = url  # Assign the URL to each face
            faces.extend(face_encodings)
        except Exception as e:
            print(f"Error processing image {url}: {e}")
    return faces

# Function to cluster faces based on similarity
def cluster_faces(faces: List[FaceCreate], clusters: List[Cluster], similarity_threshold: float = 0.6) -> List[Cluster]:
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
            modified_cluster_id.add(new_cluster.id)  # Use the new cluster's ID here

    modified_clusters = []
    for cluster in clusters:
        if cluster.id in modified_cluster_id:  # Correctly check if the cluster's ID is in the modified set
            modified_clusters.append(cluster)

    return modified_clusters

# Main execution function
def process_urls(image_urls: List[str], clusters: List[Cluster]) -> List[Cluster]:
    faces = process_images_from_urls(image_urls)
    clusters = cluster_faces(faces, clusters)
    return clusters


######################################################################
def extract_face_encoding(image) -> Optional[Face]:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = mtcnn.detect(image_rgb)
    if boxes is None:
        return None

    for box in boxes:
        # Ensure the coordinates are within the image bounds
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

        # Check if the bounding box is valid
        if x2 > x1 and y2 > y1:
            face = image_rgb[y1:y2, x1:x2]

            # Resize the face to the required size (160x160) and convert to tensor
            face_resized = cv2.resize(face, (160, 160))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float()

            # Normalize the face tensor
            face_tensor = (face_tensor - 127.5) / 128.0

            # Get the face encoding using the Facenet model
            with torch.no_grad():
                face_encoding = model(face_tensor).numpy().flatten().tolist()

            # Return the face encoding as a Face object
            return Face(embedding=face_encoding, url="")

    return None

# # Function to find the nearest cluster for a given encoding
# def find_nearest_cluster(input_encoding: List[float], clusters: List[Cluster]) -> Optional[Cluster]:
#     min_distance = float('inf')
#     nearest_cluster = None

#     # Iterate over each cluster
#     for cluster in clusters:
#         cluster_centre = np.array(cluster.cluster_centre)

#         # Calculate the distance (cosine similarity) between the input encoding and the cluster center
#         similarity = cosine_similarity([input_encoding], [cluster_centre])[0][0]
#         distance = 1 - similarity  # Convert similarity to distance

#         # If this distance is smaller, update the nearest cluster
#         if distance < min_distance:
#             min_distance = distance
#             nearest_cluster = cluster

#     return nearest_cluster

# # Function to search for similar photos within the clusters
# def search_photos(query_embedding: List[float], clusters: List[Cluster], similarity_threshold = 0.5) -> List[str]:
#     nearest_cluster = find_nearest_cluster(query_embedding, clusters)
#     similar_image_urls = set()  # Use a set to store unique URLs

#     if nearest_cluster is not None:
#         print("Searching for similar faces in the nearest cluster:")

#         # Convert the query embedding to a 2D array for cosine similarity calculation
#         query_embedding = np.array(query_embedding).reshape(1, -1)

#         for face_image in nearest_cluster.face_images:
#             encoding = np.array(face_image.embedding).reshape(1, -1)
#             image_url = face_image.url

#             # Calculate cosine similarity
#             similarity = cosine_similarity(query_embedding, encoding)[0][0]

#             # Check if the similarity is greater than 50%
#             if similarity > similarity_threshold:
#                 similar_image_urls.add(image_url)

#     else:
#         print("No similar faces found.")

#     return list(similar_image_urls)



# Function to find the nearest cluster for a given encoding
def find_similar_clusters(input_encoding: List[float], clusters: List[Cluster], threshold: float = 0.5) -> List[Cluster]:
    similar_clusters = []

    # Iterate over each cluster
    for cluster in clusters:
        cluster_centre = np.array(cluster.cluster_centre)

        # Calculate the similarity between the input encoding and the cluster center
        similarity = cosine_similarity([input_encoding], [cluster_centre])[0][0]

        # If the similarity is above the threshold, add the cluster to the list
        if similarity >= threshold:
            similar_clusters.append(cluster)

    return similar_clusters


def search_photos(query_embedding: List[float], clusters: List[Cluster], similarity_threshold: float = 0.5) -> List[str]:
    similar_clusters = find_similar_clusters(query_embedding, clusters)
    similar_image_urls = set()  # Use a set to store unique URLs

    if similar_clusters:
        # Convert the query embedding to a 2D array for cosine similarity calculation
        query_embedding = np.array(query_embedding).reshape(1, -1)

        for cluster in similar_clusters:

            for face_image in cluster.face_images:
                encoding = np.array(face_image.embedding).reshape(1, -1)
                image_url = face_image.url

                # Calculate cosine similarity
                similarity = cosine_similarity(query_embedding, encoding)[0][0]

                # Check if the similarity is greater than the threshold
                if similarity > similarity_threshold:
                    similar_image_urls.add(image_url)

    else:
        print("No similar faces found.")

    return list(similar_image_urls)

# Function to capture a face encoding from a given image URL
def capture_face_encoding(url: str) -> Optional[Face]:
    response = requests.get(url)
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return extract_face_encoding(image)

# Main execution to find similar images in clusters based on a query image
def find_images(clusters: List[Cluster], query_image_url: str) -> List[str]:
    # Capture the query face encoding from the provided image URL
    query_face = capture_face_encoding(query_image_url)

    if query_face is not None:
        return search_photos(query_face.embedding, clusters)
    else:
        return []