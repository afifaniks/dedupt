from functools import lru_cache

import torch
from dotenv import load_dotenv
from langchain.schema import Document

from methods.neural.siam.vector_manager import ChromaVectorStoreManager


class LLMEncoder:
    def __init__(self, coder, stack_formatter, multi_stack=False, bucket_name=""):
        super(LLMEncoder, self).__init__()
        self.stack_formatter = stack_formatter
        self.coder = coder
        self.multi_stack = multi_stack
        self.bucket_name = bucket_name
        self.out_dim = 1536
        load_dotenv()

        index_path = f"./{bucket_name}_chroma_index"
        meta_path = f"{bucket_name}_metadata.pkl"
        self.vector_manager = ChromaVectorStoreManager(index_path, meta_path, bucket_name, rerank=True)

    def fit(self, unsupervised_data):
        if self.vector_manager.has_index():
            print("Vectorstore already exists. Skipping fit.")
            return self.vector_manager.load_or_create([])
        
        frames_id_map = {}
        for stack_id in unsupervised_data:
            frames = self.coder(stack_id, transformer=True)
            if self.multi_stack:
                frames = [self.stack_formatter.format(frame) for frame in frames[:5]]  # Limiting to avoid token limit issues
                frames = "\n\n".join(frames)
            else:
                frames = self.stack_formatter.format(frames)
            frames_id_map[stack_id] = frames

        for id, fr in frames_id_map.items():
            # Calculate length of frames
            if len(fr) > 10000:
                print(f"Stack ID: {id}, Length of frames: {len(fr)}")
                # Truncate frames to 10000 characters
                frames_id_map[id] = fr[:10000]


        documents = [
            Document(page_content=fr, metadata={"id": id})
            for id, fr in frames_id_map.items()
        ]

        self.vector_manager.load_or_create(documents)

    # @lru_cache(maxsize=200_000)
    def forward(self, stack_id: int, candidate_ids):
        assert self.vector_manager.vectorstore is not None, "Vectorstore not initialized. Run fit() first."

        frames = self.coder(stack_id, transformer=True)
        if self.multi_stack:
            frames = [self.stack_formatter.format(frame) for frame in frames]
            content = "\n\n".join(frames)
        else:
            content = self.stack_formatter.format(frames)

        candidates = self.vector_manager.query_similar_stack_ids(content, candidate_ids)

        return candidates

    def opt_params(self) -> list:
        pass

    def out_dim(self) -> int:
        return self.out_dim

    def name(self) -> str:
        return self.bucket_name
