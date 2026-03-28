#!/usr/bin/env python3
"""
Build a UMAP visualization of the axebot corpus embeddings.
Outputs a self-contained HTML file served at /static/viz.html
"""

import time
import json
import numpy as np
from pathlib import Path

import chromadb
import umap
import plotly.graph_objects as go

DB_PATH    = Path.home() / "rag" / "db"
OUT_PATH   = Path.home() / "rag" / "static" / "viz.html"
COLLECTION = "axefx3"
SAMPLE_N   = 15000  # sample for browser performance; None = use all

SOURCE_COLORS = {
    "pdf":     "#ff6a00",
    "wiki":    "#00d4d4",
    "forum":   "#4caf50",
    "youtube": "#e53935",
}

def pull_data(col, sample_n=None):
    print("Pulling embeddings from ChromaDB...")
    all_embeddings = []
    all_metadatas  = []
    all_documents  = []
    offset = 0
    batch  = 2000
    total  = col.count()

    while offset < total:
        r = col.get(include=['embeddings','metadatas','documents'],
                    limit=batch, offset=offset)
        if not r['ids']:
            break
        all_embeddings.extend(r['embeddings'])
        all_metadatas.extend(r['metadatas'])
        all_documents.extend(r['documents'])
        offset += batch
        print(f"  {min(offset, total)}/{total}", end='\r')

    print(f"\nPulled {len(all_embeddings)} embeddings")

    embeddings = np.array(all_embeddings, dtype=np.float32)

    if sample_n and len(embeddings) > sample_n:
        print(f"Sampling {sample_n} points for browser performance...")
        idx = np.random.choice(len(embeddings), sample_n, replace=False)
        embeddings     = embeddings[idx]
        all_metadatas  = [all_metadatas[i]  for i in idx]
        all_documents  = [all_documents[i]  for i in idx]

    return embeddings, all_metadatas, all_documents

def run_umap(embeddings):
    print(f"Running UMAP on {len(embeddings)} points ({embeddings.shape[1]}D → 2D)...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings)
    print(f"UMAP done in {time.time()-t0:.1f}s")
    return coords

def pretty_label(meta):
    fname = (meta.get('filename') or '').replace('.pdf','').replace('-',' ').replace('_',' ')
    model = meta.get('model_name','')
    page  = meta.get('page')
    if model:
        return model[:80]
    if page is not None:
        return f"{fname} p.{page+1}"
    return fname[:80]

def build_plot(coords, metadatas, documents):
    print("Building Plotly figure...")
    traces = {}

    for i, (meta, doc) in enumerate(zip(metadatas, documents)):
        st    = meta.get('source_type', 'unknown')
        color = SOURCE_COLORS.get(st, '#888888')
        label = pretty_label(meta)
        text  = (doc or '')[:200].replace('<','&lt;').replace('>','&gt;')
        hover = f"<b>{label}</b><br><i>{st}</i><br><br>{text}..."

        if st not in traces:
            traces[st] = dict(x=[], y=[], text=[], marker_color=color, name=st.upper())
        traces[st]['x'].append(float(coords[i,0]))
        traces[st]['y'].append(float(coords[i,1]))
        traces[st]['text'].append(hover)

    fig = go.Figure()
    for st, t in traces.items():
        fig.add_trace(go.Scattergl(
            x=t['x'], y=t['y'],
            mode='markers',
            marker=dict(size=3, color=t['marker_color'], opacity=0.7),
            name=t['name'],
            text=t['text'],
            hovertemplate='%{text}<extra></extra>',
        ))

    fig.update_layout(
        title=dict(text='AxeBot Corpus — Embedding Space (UMAP)',
                   font=dict(color='#e0e0e0', size=16)),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0f0f0f',
        font=dict(color='#e0e0e0', family='JetBrains Mono'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(bgcolor='#141414', bordercolor='#2a2a2a', borderwidth=1),
        hovermode='closest',
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

def main():
    np.random.seed(42)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    col    = client.get_collection(COLLECTION)

    embeddings, metadatas, documents = pull_data(col, sample_n=SAMPLE_N)
    coords = run_umap(embeddings)
    fig    = build_plot(coords, metadatas, documents)

    print(f"Writing {OUT_PATH}...")
    fig.write_html(str(OUT_PATH), include_plotlyjs='cdn', full_html=True)
    print(f"Done. View at http://localhost:8503/static/viz.html")

if __name__ == '__main__':
    main()
