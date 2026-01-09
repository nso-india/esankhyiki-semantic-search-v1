



from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, json, re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

USE_QDRANT = True
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:
    USE_QDRANT = False
    faiss = None


# =========================================================
# HELPERS
# =========================================================
def clean_text(t):
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_confidence(scores, min_conf=30, max_conf=95):
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mn == mx:
        return [min_conf] * len(scores)
    return [
        round(min_conf + (s - mn) / (mx - mn) * (max_conf - min_conf), 2)
        for s in scores
    ]


# =========================================================
# IIP NORMALIZER (NEW HIERARCHY SUPPORT)
# =========================================================
def normalize_iip_filters(dataset_name, indicator_name, indicator_json):
    """
    Flattens new IIP hierarchy into standard FILTERS format
    """
    flat_filters = []
    ind_code = f"{dataset_name}_{indicator_name}"

    for base in indicator_json.get("indicators1", []):
        base_year = base.get("name")

        flat_filters.append({
            "parent": ind_code,
            "filter_name": "Base Year",
            "option": base_year
        })

        for freq in base.get("Indicator2", []):
            freq_name = freq.get("name")

            flat_filters.append({
                "parent": ind_code,
                "filter_name": "Frequency",
                "option": freq_name
            })

            for f in freq.get("filters", []):

                # Financial Year
                if "financial_year" in f:
                    for y in f["financial_year"]:
                        flat_filters.append({
                            "parent": ind_code,
                            "filter_name": "Year",
                            "option": y
                        })

                # Type → Category → SubCategory
                if "type" in f:
                    for t in f["type"]:
                        type_name = t.get("name")

                        flat_filters.append({
                            "parent": ind_code,
                            "filter_name": "Type",
                            "option": type_name
                        })

                        for cat in t.get("Category", []):
                            cat_name = cat.get("name")

                            flat_filters.append({
                                "parent": ind_code,
                                "filter_name": "Category",
                                "option": cat_name
                            })

                            for sub in cat.get("SubCategory", []):
                                flat_filters.append({
                                    "parent": ind_code,
                                    "filter_name": "SubCategory",
                                    "option": sub
                                })

    return flat_filters


# =========================================================
# LOAD PRODUCTS
# =========================================================
PRODUCTS_FILE = os.path.join("products", "products.json")
with open(PRODUCTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
    raw_products = json.load(f)

DATASETS, INDICATORS, FILTERS = [], [], []

for ds_name, ds_info in raw_products.get("datasets", {}).items():
    DATASETS.append({"code": ds_name, "name": ds_name})

    for ind in ds_info.get("indicators", []):

        ind_code = f"{ds_name}_{ind['name']}"
        INDICATORS.append({
            "code": ind_code,
            "name": ind["name"],
            "desc": ind.get("description", ""),
            "parent": ds_name
        })

        # 🔥 IIP new hierarchy
        if ds_name == "IIP" and "indicators1" in ind:
            FILTERS.extend(
                normalize_iip_filters(ds_name, ind["name"], ind)
            )

        # 🟢 All other products (old flat structure)
        else:
            for f in ind.get("filters", []):
                if isinstance(f, dict):
                    for fname, options in f.items():
                        for opt in options:
                            FILTERS.append({
                                "parent": ind_code,
                                "filter_name": fname,
                                "option": opt
                            })

print(f"[INFO] Datasets={len(DATASETS)}, Indicators={len(INDICATORS)}, Filters={len(FILTERS)}")


# =========================================================
# MODELS
# =========================================================
bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


# =========================================================
# VECTOR DB SETUP
# =========================================================
VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()
COLLECTION = "indicators_collection"

qclient = None
faiss_index = None

if USE_QDRANT:
    try:
        qclient = QdrantClient(url="http://localhost:6333")
        if COLLECTION not in [c.name for c in qclient.get_collections().collections]:
            qclient.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_DIM,
                    distance=qmodels.Distance.COSINE
                )
            )
        print("[INFO] Qdrant ready")
    except Exception as e:
        USE_QDRANT = False
        print("[WARN] Qdrant failed, using FAISS:", e)


# =========================================================
# INDEX INDICATORS
# =========================================================
names = [clean_text(i["name"]) for i in INDICATORS]
descs = [clean_text(i.get("desc", "")) for i in INDICATORS]

embeddings = (
    0.3 * bi_encoder.encode(names, convert_to_numpy=True)
    + 0.7 * bi_encoder.encode(descs, convert_to_numpy=True)
)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

if USE_QDRANT and qclient:
    qclient.upsert(
        collection_name=COLLECTION,
        points=[
            qmodels.PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload=INDICATORS[i]
            )
            for i in range(len(INDICATORS))
        ]
    )
else:
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings.astype("float32"))


# =========================================================
# SEARCH LOGIC (TOP-3 UNIQUE PRODUCTS)
# =========================================================
def search_indicators(query, top_k=25, max_products=3):
    q_vec = bi_encoder.encode([clean_text(query)], convert_to_numpy=True)
    q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True)

    candidates = []

    if USE_QDRANT and qclient:
        hits = qclient.search(
            collection_name=COLLECTION,
            query_vector=q_vec[0].tolist(),
            limit=top_k
        )
        candidates = [h.payload for h in hits]
    else:
        _, I = faiss_index.search(q_vec.astype("float32"), top_k)
        candidates = [INDICATORS[i] for i in I[0] if i >= 0]

    if not candidates:
        return []

    scores = cross_encoder.predict(
        [(query, c["name"] + " " + c.get("desc", "")) for c in candidates]
    )

    for i, c in enumerate(candidates):
        c["score"] = float(scores[i])

    candidates.sort(key=lambda x: x["score"], reverse=True)

    seen, final = set(), []
    for c in candidates:
        if c["parent"] not in seen:
            seen.add(c["parent"])
            final.append(c)
        if len(final) == max_products:
            break

    return final


# =========================================================
# FLASK APP
# =========================================================
app = Flask(__name__, template_folder="templates")
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@app.route("/search/predict", methods=["POST"])
def predict():
    q = request.json.get("query", "").strip()
    if not q:
        return jsonify({"error": "query required"}), 400

    top_results = search_indicators(q)
    confidences = normalize_confidence([r["score"] for r in top_results])

    results = []

    for ind, conf in zip(top_results, confidences):
        dataset = next(d for d in DATASETS if d["code"] == ind["parent"])
        related_filters = [f for f in FILTERS if f["parent"] == ind["code"]]

        grouped = {}
        for f in related_filters:
            grouped.setdefault(f["filter_name"], []).append(f)

        best_filters = []
        for fname, opts in grouped.items():
            pairs = [(q, f"{fname} {o['option']}") for o in opts]
            scores = cross_encoder.predict(pairs)
            best_filters.append({
                "filter_name": fname,
                "option": opts[int(np.argmax(scores))]["option"]
            })

        results.append({
            "dataset": dataset["name"],
            "indicator": ind["name"],
            "confidence": conf,
            "filters": best_filters
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5009)
