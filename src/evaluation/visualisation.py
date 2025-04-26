import plotly.express as px
import umap
import numpy as np
from sklearn.preprocessing import LabelEncoder

def plot_embeddings(text_embeddings, image_embeddings, text_labels, image_labels):
    """Visualize text and image embeddings separately in 2D space"""
    reducer = umap.UMAP(random_state=42)
    
    # Process text embeddings
    text_2d = np.empty((0, 2))
    if text_embeddings.size > 0:
        text_2d = reducer.fit_transform(text_embeddings)
    
    # Process image embeddings separately
    image_2d = np.empty((0, 2))
    if image_embeddings.size > 0:
        image_2d = reducer.fit_transform(image_embeddings)
    
    # Combine coordinates and labels
    all_x = np.concatenate([text_2d[:,0], image_2d[:,0]])
    all_y = np.concatenate([text_2d[:,1], image_2d[:,1]])
    types = ["text"] * len(text_2d) + ["image"] * len(image_2d)
    labels = text_labels + image_labels
    
    # Create plot
    fig = px.scatter(
        x=all_x, y=all_y,
        color=types, hover_name=labels,
        title="Multimodal Embedding Space",
        labels={'color': 'Embedding Type'}
    )
    return fig

def plot_search_results(query_embed, results, base_embeddings):
    """Highlight search results in embedding space"""
    reducer = umap.UMAP(random_state=42).fit(base_embeddings)
    query_2d = reducer.transform([query_embed])[0]
    result_indices = [i for i, _ in enumerate(results)]
    
    fig = px.scatter(
        x=reducer.embedding_[:,0], y=reducer.embedding_[:,1],
        title="Search Results Context"
    )
    fig.add_scatter(
        x=[query_2d[0]], y=[query_2d[1]],
        mode='markers', marker=dict(color='red', size=15),
        name='Query'
    )
    fig.add_scatter(
        x=reducer.embedding_[result_indices,0],
        y=reducer.embedding_[result_indices,1],
        mode='markers', marker=dict(color='green', size=10),
        name='Results'
    )
    return fig