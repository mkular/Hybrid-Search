import streamlit as st
from semanticsearch import SemanticSearchEngine
from sentence_transformers import util, CrossEncoder
import argparse
from utils.ssdochelper import load_config
from utils.logger_setup import log_function_call, logger
from keywordsearch import KeywordSearchEngine
import os
from utils.spotify_ep_fetcher import EpisodeFetcher
import pandas as pd
from typing import List, Tuple

class RunStreamlit:

    def __init__(self, config):
        self.config = config
        self.semanticsearch = SemanticSearchEngine(self.config)
        self.keyword_obj = KeywordSearchEngine(self.config)
        self.doc_obj = self.semanticsearch.doc_obj
        self.db_obj = self.semanticsearch.db_obj
        self.qdrant_obj = self.semanticsearch.qdrant_obj
        self.cross_encoder = CrossEncoder(self.config['CROSSENCODER_MODEL'])

    def search_candidates(self, query_str: str, chunk_df: pd.DataFrame) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        This method compares the search query vector against the semantic_chunk collection of qdrant db and
        returns top 5 candidates.
        It also returns the top 5 keyword search based candidates.
        Finally it sorts them based on a re-ranker in order to get hybrid search result.
        """
        top_sem_hits = []
        logger.info("Candidates will be produced via KNN")
        semantic_results = self.qdrant_obj.search_collection(self.semanticsearch.table_name, query_str)
        for i, hit in enumerate(semantic_results):
            result_dict = {}
            result_dict["title"] = hit.payload["name"]
            result_dict["ranking"] = 'Search Rank: {}, Relevance score: {}'.format(i, hit.score)
            result_dict["chunk_text"] = hit.payload['chunk_text']
            top_sem_hits.append(result_dict)
        top_kw_hits = []

        logger.info("Getting candidates from Keyword Search")
        kw_results, kw_scores = self.keyword_obj.run_keyword_search(chunk_df["chunk_text"], query_str)
        for i in range(kw_results.shape[1]):
            result_dict = {}
            doc, score = kw_results[0, i], kw_scores[0, i]
            ep_title, ep_chunk_txt = self.db_obj.get_from_chunk_table(doc, self.semanticsearch.table_name)
            result_dict["title"] = ep_title
            result_dict["ranking"] = 'Search Rank: {}, Relevance score: {}'.format(i, score)
            result_dict["chunk_text"] = doc
            top_kw_hits.append(result_dict)
        logger.info(f"Top hits of Keyword Search:\n {top_kw_hits}")

        logger.info("Need to perform reranking of the final results")
        rerank_hits = top_sem_hits + top_kw_hits
        cross_inp = [[query_str, res["chunk_text"]] for res in rerank_hits]
        cross_scores = self.cross_encoder.predict(cross_inp)
        logger.info(f"Cross scores are: \n {cross_scores}")
        for idx in range(len(cross_scores)):
            rerank_hits[idx]['cross-score'] = cross_scores[idx]
        rerank_hits = sorted(rerank_hits, key=lambda x: x['cross-score'], reverse=True)
        return top_sem_hits, top_kw_hits, rerank_hits

    @log_function_call
    def run_streamlit(self) -> None:
        logger.info("Preparding Data and DB for performing hybrid search")
        chu = self.semanticsearch.prepare_qdrant()
        st.title(":blue[Welcome to Hybrid Search Engine!!!]")
        search_query = st.text_input("Enter your query below:")                      
        if st.button("Search"):
            if search_query:
                ss_results, kw_search_results, reranked_results = self.search_candidates(search_query, chu)
                st.header(":blue[Semantic Search Candidates]", divider="blue")
                for result in ss_results:                             
                    st.subheader(result["title"])
                    st.write(":blue[{}]".format(result['ranking']))
                    st.write(f"[{result['chunk_text']}]")
                    st.divider()
                st.header(":blue[Keyword Search Candidates]", divider="blue")
                for result in kw_search_results:                            
                    st.subheader(result["title"])
                    st.write(":blue[{}]".format(result['ranking']))
                    st.write(f"[{result['chunk_text']}]")
                    st.divider()
                st.header(":blue[Reranked Search Candidates]", divider="blue")
                for result in reranked_results:                            
                    st.subheader(result["title"])
                    st.write(":blue[{}]".format(result['ranking']))
                    st.write(":blue[Cross-scores: {}]".format(result['cross-score']))
                    st.write(f"[{result['chunk_text']}]")
                    st.divider()  

if __name__ == "__main__":

    config_file_path = os.getcwd() + "/data/config.yaml"
    parser = argparse.ArgumentParser(description="Semantic Search Engine")    
    parser.add_argument("--config", type=str, default=config_file_path, help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    uri = config['EPISODE_URI']
    if config['CSV_FILE_NAME'] not in os.listdir(config['DATA_DIR']): 
        episodes_info = EpisodeFetcher(uri)
        logger.info("Fetching the episodes info from spotify")
        episodes_df = episodes_info.get_episodes()
        logger.info("Converting the Episode DF to a csv file")
        episodes_df.to_csv(config['DATA_DIR'] + config['CSV_FILE_NAME'])
    st_obj = RunStreamlit(config)
    logger.info("Running Streamlit App")
    st_obj.run_streamlit()
