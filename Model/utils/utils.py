import math

def split_tasks(df, n_chunks):
    chunk_size = math.floor(len(df) / n_chunks)
    chunks = [
        df[chunk_size * i : chunk_size * (i + 1)] if i < n_chunks - 1 else df[chunk_size * i :]
        for i in range(n_chunks)
    ]
    return chunks
