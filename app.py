import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import ast
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Seattle Stay Finder",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --cream: #F5F0E8;
    --forest: #2C4A3E;
    --sage: #7A9E8E;
    --rust: #C4622D;
    --warm-white: #FDFBF7;
    --charcoal: #2A2A2A;
    --light-sage: #E8F0EC;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--warm-white);
    color: var(--charcoal);
}

.main { background-color: var(--warm-white); }

h1, h2, h3 { font-family: 'Fraunces', serif; }

.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 3.2rem;
    font-weight: 300;
    color: var(--forest);
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: var(--sage);
    font-weight: 300;
    margin-bottom: 2rem;
}

.listing-card {
    background: white;
    border-radius: 16px;
    padding: 0;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(44,74,62,0.08);
    overflow: hidden;
    transition: box-shadow 0.2s;
    border: 1px solid rgba(44,74,62,0.08);
}
.listing-card:hover {
    box-shadow: 0 6px 24px rgba(44,74,62,0.14);
}
.card-img {
    width: 100%;
    height: 180px;
    object-fit: cover;
}
.card-body {
    padding: 1rem 1.2rem 1.2rem;
}
.card-title {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: var(--forest);
    margin-bottom: 0.3rem;
}
.card-meta {
    font-size: 0.82rem;
    color: #888;
    margin-bottom: 0.5rem;
}
.card-price {
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--rust);
}
.card-rating {
    font-size: 0.85rem;
    color: var(--sage);
}
.card-reason {
    font-size: 0.83rem;
    color: #555;
    background: var(--light-sage);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-top: 0.7rem;
    line-height: 1.5;
    border-left: 3px solid var(--sage);
}
.similarity-badge {
    display: inline-block;
    background: var(--forest);
    color: white;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 6px;
    vertical-align: middle;
}
.superhost-badge {
    display: inline-block;
    background: var(--rust);
    color: white;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
}
.stButton > button {
    background-color: var(--forest) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background-color: var(--rust) !important;
}
.filter-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--forest);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--forest) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.result-count {
    font-family: 'Fraunces', serif;
    font-size: 1.1rem;
    color: var(--forest);
    margin-bottom: 1rem;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ── Load Data & Models ────────────────────────────────────────
@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer
    from groq import Groq
    import os
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=groq_api_key)
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    return client, model

@st.cache_data
def load_data():
    # Load CSV from Google Drive
    file_id = "1eNPJJztqgoPc1-yAJoD34ua1RoHR3gX7"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    listings = pd.read_csv(url)
    
    # Load npy from GitHub Repository
    embeddings = np.load('listing_embeddings.npy')
    
    listings['price_clean'] = listings['price'].replace(r'[\$,]', '', regex=True).astype(float)
    listings = listings.reset_index(drop=True)
    return listings, embeddings

client, embed_model = load_models()
listings, embeddings = load_data()


# ── Core Functions ────────────────────────────────────────────
def parse_query(user_input, room_type=None, superhost=None, max_price=None, neighbourhood=None):
    filters_context = ""
    if room_type and room_type != "Any":
        filters_context += f"\n- Room type is already set to: {room_type}"
    if superhost:
        filters_context += f"\n- Superhost filter is ON"
    if max_price:
        filters_context += f"\n- Max price is already set to: ${max_price}"
    if neighbourhood and neighbourhood != "Any":
        filters_context += f"\n- Neighbourhood is already set to: {neighbourhood}"

    prompt = f"""
You are helping parse a user's Airbnb search query.
Extract search conditions from the user input and return ONLY a JSON object, no explanation.

User input: "{user_input}"
Already set filters (do not override):{filters_context if filters_context else " none"}

Return JSON with these fields (use null if not mentioned, do not override already set filters):
{{
    "room_type": "Entire home/apt" or "Private room" or "Shared room" or null,
    "neighbourhood": string or null,
    "min_price": number or null,
    "max_price": number or null,
    "superhost": true or false or null,
    "min_rating": number or null,
    "amenities_keywords": [list of strings] or [],
    "search_text": "cleaned search intent for embedding"
}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    text = response.choices[0].message.content.strip()
    try:
        text = text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"search_text": user_input, "amenities_keywords": []}

    # Apply UI filters (override if set)
    if room_type and room_type != "Any":
        parsed['room_type'] = room_type
    if superhost:
        parsed['superhost'] = True
    if max_price:
        parsed['max_price'] = max_price
    if neighbourhood and neighbourhood != "Any":
        parsed['neighbourhood'] = neighbourhood

    return parsed


def search_listings(user_input, parsed, top_k=10):
    mask = np.ones(len(listings), dtype=bool)

    if parsed.get('room_type'):
        mask &= (listings['room_type'] == parsed['room_type']).values
    if parsed.get('superhost') is True:
        mask &= (listings['host_is_superhost'] == 't').values
    if parsed.get('min_price'):
        mask &= (listings['price_clean'] >= parsed['min_price']).values
    if parsed.get('max_price'):
        mask &= (listings['price_clean'] <= parsed['max_price']).values
    if parsed.get('neighbourhood'):
        mask &= listings['neighbourhood_cleansed'].str.contains(
            parsed['neighbourhood'], case=False, na=False
        ).values
    if parsed.get('min_rating'):
        mask &= (listings['review_scores_rating'] >= parsed['min_rating']).values

    filtered_listings = listings[mask].reset_index(drop=True)
    filtered_embeddings = embeddings[mask]

    if len(filtered_listings) == 0:
        return None

    search_text = parsed.get('search_text', user_input)
    query_embedding = embed_model.encode([search_text], normalize_embeddings=True)
    similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]

    top_indices = similarities.argsort()[::-1][:top_k]
    results = filtered_listings.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    return results


def generate_reason(user_input, row, parsed):
    matched = []
    if parsed.get('max_price') and row.get('price_clean', float('inf')) <= parsed['max_price']:
        matched.append(f"under ${parsed['max_price']}")
    if parsed.get('superhost') and row.get('host_is_superhost') == 't':
        matched.append("superhost")
    if parsed.get('neighbourhood'):
        matched.append(f"near {parsed['neighbourhood']}")
    if parsed.get('room_type'):
        matched.append(parsed['room_type'])

    matched_str = ', '.join(matched) if matched else 'general preferences'
    desc = row.get('description')
    desc_text = str(desc)[:300] if pd.notna(desc) else 'No description available'

    prompt = f"""
You are a helpful Airbnb assistant. Explain why this listing is a good match in 2 sentences max. Be specific and friendly.

User's request: "{user_input}"
Matched conditions: {matched_str}

Listing:
- Name: {row.get('name', '')}
- Neighbourhood: {row.get('neighbourhood_cleansed', '')}
- Price: {row.get('price', '')}
- Rating: {row.get('review_scores_rating', '')}
- Description: {desc_text}

2 sentences max, reference matched conditions.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# ── UI Layout ─────────────────────────────────────────────────
st.markdown('<div class="hero-title">Find your<br><i>perfect stay</i> in Seattle.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered search across thousands of listings</div>', unsafe_allow_html=True)

# Search bar
search_input = st.text_input(
    label="search",
    placeholder='e.g. "Cozy entire home near Capitol Hill under $150, superhost preferred"',
    label_visibility="collapsed"
)

# Filters row
col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1.5, 1])

with col1:
    room_type_options = ["Any", "Entire home/apt", "Private room", "Shared room"]
    room_type = st.selectbox("Room Type", room_type_options)

with col2:
    neighbourhoods = ["Any"] + sorted(listings['neighbourhood_cleansed'].dropna().unique().tolist())
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)

with col3:
    max_price = st.slider("Max Price / night", 0, 1000, 300, step=10, format="$%d")

with col4:
    superhost = st.checkbox("Superhost only", value=False)

with col5:
    top_k = st.selectbox("Results", [5, 10, 15, 20], index=1)

search_clicked = st.button("Search →", use_container_width=False)

# ── Search Execution ──────────────────────────────────────────
if search_clicked and search_input.strip():
    with st.spinner("Finding your perfect stay..."):
        parsed = parse_query(
            search_input,
            room_type=room_type if room_type != "Any" else None,
            superhost=superhost if superhost else None,
            max_price=max_price if max_price > 0 else None,
            neighbourhood=neighbourhood if neighbourhood != "Any" else None
        )
        results = search_listings(search_input, parsed, top_k=top_k)

    if results is None or len(results) == 0:
        st.warning("No listings found. Try relaxing your filters.")
    else:
        st.markdown(f'<div class="result-count">Found {len(results)} listings for you.</div>', unsafe_allow_html=True)

        # Map + Cards layout
        map_col, cards_col = st.columns([1.2, 1])

        with map_col:
            map_data = results[['latitude', 'longitude', 'name']].dropna()
            st.map(map_data, zoom=12, use_container_width=True)

        with cards_col:
            st.markdown("#### Top Results")
            for i, (_, row) in enumerate(results.iterrows()):
                with st.container():
                    img_url = row.get('picture_url', '')
                    name = row.get('name', 'Unnamed')
                    hood = row.get('neighbourhood_cleansed', '')
                    price = row.get('price', '')
                    rating = row.get('review_scores_rating', '')
                    is_superhost = row.get('host_is_superhost') == 't'
                    sim = row.get('similarity_score', 0)
                    listing_url = row.get('listing_url', '#')
                    room = row.get('room_type', '')
                    beds = row.get('beds', '')
                    baths = row.get('bathrooms_text', '')

                    superhost_badge = '<span class="superhost-badge">⭐ Superhost</span>' if is_superhost else ''
                    sim_badge = f'<span class="similarity-badge">{sim:.0%} match</span>'

                    reason_html = ""
                    if i < 3:
                        with st.spinner(f"Generating reason for #{i+1}..."):
                            reason = generate_reason(search_input, row, parsed)
                        reason = reason.replace('"', '&quot;').replace("'", '&#39;')
                        reason_html = f'<div class="card-reason">💬 {reason}</div>'

                    card_html = f"""
<div class="listing-card">
    {'<img class="card-img" src="' + img_url + '" onerror="this.style.display=\'none\'">' if img_url else ''}
    <div class="card-body">
        <div class="card-title">{name} {sim_badge}</div>
        <div class="card-meta">📍 {hood} &nbsp;·&nbsp; {room} &nbsp;·&nbsp; 🛏 {beds} beds &nbsp;·&nbsp; 🚿 {baths}</div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <span class="card-price">{price}</span>
            <span class="card-rating">⭐ {rating}</span>
            {superhost_badge}
        </div>
        {reason_html}
        <div style="margin-top:0.8rem;">
            <a href="{listing_url}" target="_blank" style="font-size:0.82rem; color:var(--forest); font-weight:500; text-decoration:none; border-bottom: 1px solid var(--sage);">View on Airbnb →</a>
        </div>
    </div>
</div>
"""
                    st.markdown(card_html, unsafe_allow_html=True)

elif search_clicked and not search_input.strip():
    st.info("Please enter a search query.")
