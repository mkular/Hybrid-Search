import docker.errors
from qdrant_client import models, QdrantClient
from utils.logger_setup import logger
import docker
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import ScrollRequest
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import pandas as pd

class QdrantHelper:
    """
    Helper methods to interacting with qdrant db.
    """
    def __init__(self, config):
        self.client = docker.from_env()
        self.qclient = QdrantClient(host="localhost", port=6333)
        self.db_path = config['qdrant_path']
        self.top_k = config['semantic_top_k']
        self.model = SentenceTransformer(config["MODEL"])

    def start_qdrant_container(self):
        try:
            container = self.client.containers.get("qdrant_container")
            if container.status == "running":
                print("Qdrant container is already running.")
            else:
                container.start()
                print("Qdrant container started.")
        except docker.errors.NotFound:
            # Create and start a new Qdrant container if not found
            container = self.client.containers.run(
                "qdrant/qdrant",
                name="qdrant_container",
                ports={"6333/tcp": 6333},
                volumes={self.db_path: {"bind": "/qdrant/storage", "mode": "rw"}},
                detach=True
            )
            print("New Qdrant container started.")


    # Function to stop the Qdrant container
    def stop_qdrant_container(self):
        try:
            container = self.client.containers.get("qdrant_container")
            if container.status == "running":
                container.stop()
                print("Qdrant container stopped.")
            else:
                print("Qdrant container is already stopped.")
        except docker.errors.NotFound:
            print("Qdrant container not found.")


    def create_collection(self, collection_name):
        try:
            self.qclient.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    distance=Distance.COSINE,
                    size=self.model.get_sentence_embedding_dimension(),
                    ), # Options: "Cosine", "Euclid", "Dot"
            )
        except Exception as e:
            logger.error(f"An error occured while creating qdrant table: {e}")

    def upload_points(self, collection_name, chunk_df):
        self.qclient.upload_points(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=idx, vector=self.model.encode(row["description"]).tolist(), payload=row
                )
                for idx, row in enumerate(chunk_df.to_dict(orient="records"))
            ],
        )

    def search_collection(self, collection_name, query):
        search_result = self.qclient.query_points(
            collection_name=collection_name,
            query=self.model.encode(query).tolist(),
            limit=self.top_k  # Return top 3 results
        ).points
        return search_result
        """
        for hit in search_result:
            print(f"ID: {hit.id}, Score: {hit.score}, Metadata: {hit.payload}")
        """

    def check_collection_exists(self, collection_name):
        try:
            self.qclient.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' exists.")
            return True
        except UnexpectedResponse:
            logger.info(f"Collection '{collection_name}' does not exist.")
            return False
        
    def delete_collection(self, collection_name):
        try:
            self.qclient.delete_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' successfully delete")
        except Exception as e:
            logger.info(f"Collection '{collection_name}' not found. Error: {e}")
            return False
        
    
    def get_vector_list(self, collection_name):
        try:
                    # Initialize an empty list to store all vector embeddings
            all_embeddings = []

            # Set the batch size for scrolling (adjust as needed)
            batch_size = 100

            # Scroll through all points
            offset = None
            while True:
                response, point_id = self.qclient.scroll(
                    collection_name=collection_name,
                    limit=batch_size, 
                    offset=offset,
                    with_payload=False,  # We don't need payloads
                    with_vectors=True     # We want vectors
                )
                
                # Extract vectors from the response
                for point in response:
                    all_embeddings.append(point.vector)
                
                # Check if we've retrieved all points
                if len(response) < batch_size:
                    break
                
                # Update the offset for the next batch
                offset = point_id
            return all_embeddings
        except UnexpectedResponse:
            logger.info(f"Unable to get data from '{collection_name}'.")
            return False
