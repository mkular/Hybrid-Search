import pandas as pd
import sqlite3
from utils.logger_setup import log_function_call,logger

class DbHelper:
    """
    Helper methods for interacting with sqlite db.
    """
    def __init__(self, db_name, file_path):
        self.db_name = db_name
        self.file_path = file_path

    def populate_db(self, table_name: str, df: pd.DataFrame):
        conn = sqlite3.connect(self.file_path, timeout=5)
        #Make sure document tables is present first
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()

    def get_from_chunk_table(self, value, table_name):
        conn = sqlite3.connect(self.file_path, timeout=5)
        cursor = conn.cursor()
        try:
            #Make sure document tables is present first
            get_query = f'SELECT * FROM {table_name} WHERE chunk_text = ?'
            query_data = cursor.execute(get_query, (value,))
            get_data = query_data.fetchone()
            name, chunk_txt =  get_data[14], get_data[2]
            conn.close()
            return name, chunk_txt
        except sqlite3.Error as e:
            logger.error(f"An error occurred: {e}")
    
    def dump_table(self, table_name):
        conn = sqlite3.connect(self.file_path, timeout=5)
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except sqlite3.Error as e:
            logger.error(f"An error occurred: {e}")        
