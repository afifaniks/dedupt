import os
import pickle
from typing import List, Tuple

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from methods.neural.siam.reranker import rerank_with_llm


class ChromaVectorStoreManager:
    def __init__(
        self,
        persist_directory: str,
        meta_path: str,
        bucket_name: str,
        embedding_model_name: str = "text-embedding-3-small",
        rerank: bool = False
    ):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.meta_path = os.path.join(persist_directory, meta_path)
        self.bucket_name = bucket_name
        self.embedding_client = OpenAIEmbeddings(model=embedding_model_name)
        self.vectorstore = None
        self.metadata = []
        self.rerank = rerank

    def _add_documents_in_batches(self, documents: List[Document], batch_size: int = 32):
        print(f"Adding {len(documents)} documents in batches of {batch_size}...")
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding documents"):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)

    def load_or_create(self, documents: List[Document]):
        if self.has_index():
            print(f"Loading Chroma index from {self.persist_directory}...")
            self.vectorstore = Chroma(
                collection_name=f"{self.bucket_name}_chroma_collection",
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_client
            )
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            print("Creating new Chroma index...")
            self.vectorstore = Chroma(
                collection_name=f"{self.bucket_name}_chroma_collection",
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_client
            )
            self._add_documents_in_batches(documents, batch_size=32)
            self.metadata = [doc.metadata for doc in documents]
            self.persist()

    def has_index(self) -> bool:
        index_file = os.path.join(self.persist_directory, "chroma.sqlite3")
        return os.path.exists(index_file) and os.path.exists(self.meta_path)

    def has_id(self, stack_id: int) -> bool:
        return any(m.get("id") == stack_id for m in self.metadata)

    def add_document(self, stack_id: int, content: str):
        new_doc = Document(page_content=content, metadata={"id": stack_id})
        self.vectorstore.add_documents([new_doc])
        self.metadata.append({"id": stack_id})
        self.persist()

    def persist(self):
        # Removed self.vectorstore.persist() because Chroma handles it internally
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def query_similar_stack_ids(
        self, 
        query_stack: str, 
        candidate_ids: List[int], 
        k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Return top-k similar stack trace IDs (and scores) among the specified candidate_ids.
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call `load_or_create()` first.")
        
        # Check if all candidate_ids are present in metadata
        if not all(self.has_id(stack_id) for stack_id in candidate_ids):
            print("Some candidate IDs are not present in the vector store metadata.")

        # Use metadata filter: "id" must be in candidate_ids
        # Do filtered similarity search
        query_embedding = self.embedding_client.embed_query(query_stack)
        print("Filtered candidate ID count:", len(candidate_ids))

        results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
            embedding=query_embedding,
            k=k,
            filter={"id": {"$in": candidate_ids}}
        )

        # Results already come with similarity scores between 0 and 1
        if not self.rerank:
            return [(doc.metadata["id"], score) for doc, score in results]
        
        results = [(doc.metadata["id"], score, doc.page_content) for doc, score in results]

        reranked_results = rerank_with_llm(query_stack, results)

        return reranked_results

