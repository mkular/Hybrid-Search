import pandas as pd
import re
from sentence_transformers import util
from utils.logger_setup import log_function_call, logger
import yaml

class DocumentProcessor:
    """
    Helper methods for dealing with docs like reading file,
    loading csv file and creating chunks.
    """
    def __init__(self, data_dir, transript_dir, model):
        self.data_dir = data_dir
        self.transcript_dir = transript_dir
        self.model = model

    @staticmethod
    def read_file(file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except Exception as e:
            logger.error(f"An error occured while reading file: {e}")
            return None

    def load_csv(self, file_name):
        csv_path = self.data_dir + file_name
        try:
            csv_df = pd.read_csv(csv_path)
            return csv_df
        except FileNotFoundError:
            logger.info(f"File not found at {csv_path}")
            return None
        except Exception as e:
            logger.error(f"An error occured while loading csv file: {e}")
            return None
    
    @log_function_call
    def get_chunks(self, file_name, embeddings="", min_similarity=0.5):
        doc_obj = self.load_csv(file_name)
        final_list = []
        chunk_id = 1
        doc_id = 1
        for index, row in doc_obj.iterrows():
            sentence_splitter = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
            paragraph = re.split(sentence_splitter, row['description'])
            if embeddings != "":
                chunked_text = []
                current_chunk = []
                for i, sentence in enumerate(paragraph):
                    logger.info(f"i is {i}, sentence is {sentence}")
                    current_chunk.append(sentence)
                    logger.info(f"len(paragraph) is {len(paragraph)}")
                    if i == len(paragraph) - 1 or util.cos_sim(embeddings[i], embeddings[i+1]) < min_similarity:
                        chunked_text.append(' '.join(current_chunk))
                        current_chunk = []
                paragraph = chunked_text
            for line in paragraph:
                new_row = row.copy()
                new_row["chunk_text"] = line
                new_row["chunk_id"] = chunk_id
                new_row["doc_id"] = doc_id
                chunk_id += 1
                final_list.append(new_row)            
            doc_id += 1
        chunk_df = pd.DataFrame(final_list)
        return chunk_df

@log_function_call
def load_config(config_path: str) -> dict:
    """
    Loading config.yaml into a python dict.
    """
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config
    except FileNotFoundError:
        return "The configuration file was not found."
    except yaml.YAMLError as exc:
        return f"Error in configuration file: {exc}"