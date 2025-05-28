import json

from pyserini.search.lucene import LuceneSearcher
from rank_bm25 import BM25Okapi
from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

# 使用绝对路径加载索引，确保路径正确
searcher = LuceneSearcher('wikipedia-kilt-doc-20210421')


class LocalFileBM25Retriever(BaseRetriever):
    file_path: str = Field(...)
    k: int = Field(default=5)
    documents: List[Document] = Field(default_factory=list)
    bm25: BM25Okapi = Field(default=None)  # 给一个默认值，避免 Pydantic 报错

    def __init__(self, file_path: str, k: int = 5):
        # 初始化空模型，避免传入未就绪字段
        super().__init__(file_path=file_path, k=k)

        # 加载文档并更新字段
        docs = self._load_documents(file_path)
        object.__setattr__(self, 'documents', docs)

        # 构建 BM25 模型
        corpus = [doc.page_content.split() for doc in docs]
        bm25 = BM25Okapi(corpus)
        object.__setattr__(self, 'bm25', bm25)

    def _load_documents(self, file_path: str) -> List[Document]:
        docs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(Document(page_content=line))
        return docs

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:self.k]
        return [self.documents[i] for i in top_indices]


retriever = LocalFileBM25Retriever("N-ary knowledge graph/n_ary_kg_text.txt", k=1)


def search_wiki(query):
    list_search_item = searcher.search(query)
    if len(list_search_item) == 0:
        return ""
    else:
        doc_id = list_search_item[0].docid
        words = json.loads(searcher.doc(doc_id).raw())['contents'].split()
        if len(words) >= 300:
            words_top300 = words[:300]
        else:
            words_top300 = words
        doc_search = ' '.join(words_top300)
        return doc_search


def search_kg(query):
    result = retriever.invoke(query)
    if len(result) > 0:
        return result[0].page_content
    else:
        return ""

# search_kg("Donald Trump")
