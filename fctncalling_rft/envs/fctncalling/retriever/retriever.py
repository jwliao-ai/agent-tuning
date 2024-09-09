import numpy as np
import json
import yaml
import faiss
from tqdm import tqdm
from typing import Dict, Union, Callable, List
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        embedding_model: Union[str, SentenceTransformer],
        file_path: str,
        index_func: Union[str, Callable[Dict, str]] = lambda item: item["description"],
        load_type: str = "from_file",
    ):  # type: ignore

        self.embedding_model = (
            embedding_model
            if isinstance(embedding_model, SentenceTransformer)
            else SentenceTransformer(embedding_model, trust_remote_code=True)
        )
        self.show_progress_bar = True
        self.file_path = file_path

        if isinstance(index_func, str):
            self.index_func = eval(index_func)
        else:
            self.index_func = index_func

        self._load_data(load_type)

    def _load_data(self, load_type: str):
        if load_type == "from_file":
            self._load_from_file()
        else:
            raise Exception("load setting type error")
        print(f"Retriever: data loaded successfully.")

    def _load_from_file(self):
        self.item_list = (
            json.load(open(self.file_path, encoding="utf-8"))
            if self.file_path.endswith("json")
            else yaml.safe_load(open(self.file_path, encoding="utf-8"))
        )
        texts = []
        self.datasets = []
        for item in tqdm(self.item_list, desc=f"Loading {self.file_path}"):
            texts.append(self.index_func(item))
            self.datasets.append(item)
        texts_embeddings = self.embedding_model.encode(
            texts, show_progress_bar=self.show_progress_bar, normalize_embeddings=True
        )
        d = texts_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(texts_embeddings)

    def get_relevant_documents(self, query: str, k: int = 10):
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        _, indices = self.index.search(query_embedding, k)
        relevant_documents = [self.datasets[idx.item()] for idx in indices[0][:k]]
        return relevant_documents

    # def _load_from_sql(self, sqluri: str, table: str, **kwargs):
    #     from sqlalchemy import create_engine, text
    #     from sqlalchemy.orm import sessionmaker

    #     columns = ["name", "title", "plugin", "description", "parameters", "required", "outputs"]
    #     sql_text = f"SELECT {', '.join(columns)} FROM {table} WHERE enabled=1"
    #     logger.info(f"sql_text: {sql_text}")

    #     def load_data():
    #         with sessionmaker(create_engine(sqluri, echo=False))() as session:
    #             data_cursor = session.execute(text(sql_text))
    #             for row in data_cursor:
    #                 item = dict(zip(columns, row))
    #                 for key in item.keys():
    #                     try:
    #                         item[key] = json.loads(item[key])
    #                     except:
    #                         pass
    #                 yield item
    #     item_list = load_data()
    #     self._load(item_list, **kwargs)
