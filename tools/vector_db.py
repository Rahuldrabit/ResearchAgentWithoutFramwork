import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_DB_PATH, COLLECTION_NAME
from models.data_models import ResearchPaper

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

def store_paper(paper: ResearchPaper):
    """Store a paper in the vector database"""
    collection.add(
        ids=[paper.id],
        documents=[paper.abstract],
        metadatas=[{
            "title": paper.title,
            "authors": ", ".join(paper.authors),
            "url": paper.url,
            "publication_date": paper.publication_date,
            "keywords": ", ".join(paper.keywords)
        }]
    )

def search_papers(query: str, limit: int = 5):
    """Search for papers in the vector database"""
    results = collection.query(
        query_texts=[query],
        n_results=limit
    )
    
    papers = []
    for i in range(len(results['ids'][0])):
        paper_id = results['ids'][0][i]
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]
        
        papers.append({
            "id": paper_id,
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", "").split(", "),
            "abstract": document,
            "url": metadata.get("url", ""),
            "publication_date": metadata.get("publication_date", ""),
            "keywords": metadata.get("keywords", "").split(", ")
        })
    
    return papers

def retrieve_context_for_question(question: str, limit: int = 3, min_score: float = 0.2):
    """
    Retrieve relevant context for a question from the vector database.
    Returns both documents and their metadata.
    Compatible with older versions of ChromaDB.
    """
    # Check if we can get distances from the query
    try:
        # Try with newer ChromaDB API
        results = collection.query(
            query_texts=[question],
            n_results=limit,
            include_distances=True
        )
        has_distances = True
    except TypeError:
        # Fall back to older ChromaDB API
        results = collection.query(
            query_texts=[question],
            n_results=limit
        )
        has_distances = False
    
    # Prepare context with documents and relevance
    contexts = []
    
    if not results['ids'][0]:  # No results found
        return []
    
    for i in range(len(results['ids'][0])):
        paper_id = results['ids'][0][i]
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]
        
        # Assign a dummy score if distances aren't available
        if has_distances:
            score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
        else:
            # Use position as a proxy for relevance when distances aren't available
            score = 1.0 - (i / max(1, len(results['ids'][0])))
        
        # Skip if below relevance threshold
        if score < min_score:
            continue
            
        contexts.append({
            "id": paper_id,
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", "").split(", "),
            "abstract": document,
            "relevance_score": score,
            "url": metadata.get("url", "")
        })
    
    # Sort by relevance score (will maintain original order if using position-based scores)
    contexts.sort(key=lambda x: x["relevance_score"], reverse=True)
    return contexts
