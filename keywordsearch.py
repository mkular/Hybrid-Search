import bm25s
import Stemmer
from utils.ssdbhelper import DbHelper
from utils.logger_setup import log_function_call, logger
from typing import List, Tuple

class KeywordSearchEngine:

    def __init__(self, config: dict) -> None:
        self.config = config
        self.db_obj = DbHelper(self.config['db_name'], self.config['db_path'])
        self.table_name = "chunk"
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = bm25s.BM25()
        self.top_k = config['keyword_top_k']

    @log_function_call
    def run_keyword_search(self, corpus: List[str], query: str) -> Tuple[Tuple, Tuple]:
        """
        Using bm25s create corpus tokens. 
        Create query token via the same way.
        Perform keyword search and retrieve the results, scores.
        """       
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=self.stemmer)
        self.retriever.index(corpus_tokens)
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(query_tokens, corpus=corpus, k=self.top_k)
        logger.info(results)
        logger.info(scores)
        return results, scores

