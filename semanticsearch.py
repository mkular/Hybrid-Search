import os
from utils.ssdochelper import DocumentProcessor
from utils.ssdbhelper import DbHelper
from utils.logger_setup import log_function_call, logger
from utils.qdranthelper import QdrantHelper
import time

class SemanticSearchEngine:

    def __init__(self, config):
        self.config = config
        self.doc_obj = DocumentProcessor(self.config['DATA_DIR'], self.config['TRANSCRIPT_DIR'], self.config['MODEL'])
        self.db_obj = DbHelper(self.config['db_name'], self.config['db_path'])
        self.table_name = "semantic_chunks"
        self.qdrant_obj = QdrantHelper(self.config)

    def prepare_data(self, embeddings=""):
        """
        Create brute force chunks from csv file.
        2nd pass of this method creates semantic chunks.
        """
        chunk_df = self.doc_obj.get_chunks(self.config['CSV_FILE_NAME'])
        ##Update chunk_df with chunk_embeddings column
        if embeddings != "":
            chunk_df = self.doc_obj.get_chunks(self.config['CSV_FILE_NAME'], embeddings)
        return chunk_df
        
    def prepare_db(self, chunk_df, collection_name="", qdrant=False):
        """
        Used to store data in qdrant db or sqlite db.
        """
        try:
            #DB related code
            if qdrant:
                logger.info("Starting qdrant container")
                self.qdrant_obj.start_qdrant_container()
                logger.info("Creating chunk collection")
                self.qdrant_obj.create_collection(collection_name)
                logger.info("Adding chunk embeddings and chunk metadata")
                self.qdrant_obj.upload_points(collection_name, chunk_df)
            else:
                df_dict = {self.table_name: chunk_df}
                #Create schema if DB doesnt exist
                if self.config['db_name'] not in os.listdir(self.config['DATA_DIR']):
                    #logger.info("DB does not exist. Creating db schema")
                    #self.db_obj.create_schema()
                    for name, table_df in df_dict.items():
                        logger.info("Populating DB tables")
                        self.db_obj.populate_db(name, table_df)
        except Exception as e:
            logger.error(f"prepare_db failed due to: {e}")      

    def get_existing_data(self):
        try:
            chunk_df = self.db_obj.dump_table(self.table_name)
            return chunk_df
        except Exception as e:
            logger.error(f"Error while getting data from sqlite table: {e}")

    def files_present(self):
        required_files = [self.config['db_path']]
        return all(os.path.isfile(os.path.join(self.config['BASE_DIR'], file)) for file in required_files)
    
    def prepare_qdrant(self):
        """
        This is the main method which calls all other methods in the class.
        It mainly creates qdrant collections (bute_chunks and semantic_chunks) and sqlite db table.
        """
        if self.qdrant_obj.check_collection_exists(self.table_name) and self.files_present():
            logger.info("Collection already exists. No need to add data")
            logger.info("Getting collection data")
            chunk_df = self.get_existing_data()
            embedded_chunks = self.qdrant_obj.get_vector_list(collection_name="semantic_chunks")
        else:           
            chunk_df= self.prepare_data()
            logger.info("Creating brute_chunks Collection in Qdrant")
            self.prepare_db(chunk_df, collection_name="brute_chunks", qdrant=True)
            time.sleep(5)
            logger.info("Getting list of vectors for performing semantic chunking")
            embedded_chunks = self.qdrant_obj.get_vector_list(collection_name="brute_chunks")
            sem_chunk_df = self.prepare_data(embedded_chunks)
            logger.info("Creating semantic_chunks Collection in Qdrant")
            self.prepare_db(sem_chunk_df, collection_name="semantic_chunks", qdrant=True)
            logger.info("Preparing sqlite with the semantic chunk data")
            self.prepare_db(sem_chunk_df)
        #temp for handling errors
        return chunk_df
