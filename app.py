import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login
import time

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
    --grey: #555555;
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
    border: 1px solid rgba(44,74,62,0.08);
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
    color: #2C4A3E;
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
    color: #555555;
}
.card-rating {
    font-size: 0.85rem;
    color: #7A9E8E;
}
.card-reason {
    font-size: 0.83rem;
    color: #555;
    background: #E8F0EC;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-top: 0.7rem;
    line-height: 1.5;
    border-left: 3px solid #7A9E8E;
}
.similarity-badge {
    display: inline-block;
    background: #2C4A3E;
    color: white;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 6px;
    vertical-align: middle;
}
.superhost-badge {
    display: inline-block;
    background: #555555;
    color: white;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
}
.stButton > button {
    background-color: #2C4A3E !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background-color: #555555 !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stSlider"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #2C4A3E !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.result-count {
    font-family: 'Fraunces', serif;
    font-size: 1.1rem;
    color: #2C4A3E;
    margin-bottom: 1rem;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ── Load Data & Models ────────────────────────────────────────
@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    return client, model

@st.cache_data
def load_data():
    login(token=st.secrets["HF_TOKEN"])
    listings = pd.read_csv("hf://datasets/keeljimin/HouseSearchAndDiscovery/listings_processed.csv")
    embeddings = np.load('listing_embeddings.npy')
    listings['price_clean'] = listings['price'].replace(r'[\$,]', '', regex=True).astype(float)
    listings = listings.reset_index(drop=True)
    return listings, embeddings

client, embed_model = load_models()
listings, embeddings = load_data()


# ── Core Functions ────────────────────────────────────────────
def extract_search_text(user_input):
    prompt = f"""
You are a Seattle local helping match Airbnb listings.

MOST IMPORTANT: Extract location first. If a specific neighborhood or landmark is mentioned, it must appear first in the output.

Then consider: property type, amenities, safety, vibe, and trip purpose (tourism/business/budget/visiting friends).
Expand Seattle abbreviations (SLU -> South Lake Union, Cap Hill -> Capitol Hill, etc.)

Output format: "[LOCATION] [property type] [trip purpose/vibe] [amenities]"
15-20 words max, no explanation.

Input: "{user_input}"
Output:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        st.warning("🤖 AI is a little busy right now. Please try again in a moment!")
        st.stop()
    
def search_listings(user_input, room_type=None, superhost=None, max_price=None, neighbourhood=None, top_k=10):
    # 하드 필터: UI에서 선택한 것만
    mask = np.ones(len(listings), dtype=bool)

    if room_type:
        mask &= (listings['room_type'].isin(room_type)).values
    if superhost:
        mask &= (listings['host_is_superhost'] == 't').values
    if min_price and min_price > 0:
        mask &= (listings['price_clean'] >= min_price).values
    if max_price and max_price < 1000:
        mask &= (listings['price_clean'] <= max_price).values
    if min_rating:
        mask &= (listings['review_scores_rating'] >= min_rating).values
    if neighbourhood:
        mask &= listings['neighbourhood_group_cleansed'].isin(neighbourhood).values

    filtered_listings = listings[mask].reset_index(drop=True)
    filtered_embeddings = embeddings[mask]

    # 필터 결과 없으면 전체에서 검색
    if len(filtered_listings) == 0:
        filtered_listings = listings.reset_index(drop=True)
        filtered_embeddings = embeddings

    # 유사도 검색
    search_text = extract_search_text(user_input)
    query_embedding = embed_model.encode([search_text], normalize_embeddings=True)
    similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]

    top_indices = similarities.argsort()[::-1][:top_k]
    results = filtered_listings.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    return results


def generate_reason(user_input, row, matched_filters):
    # Extract only filters the user explicitly set (ignore None values)
    active_filters = {k: v for k, v in matched_filters.items() if v is not None}
    
    # Build met/unmet condition lists based on active filters only
    met, unmet = [], []
    
    if active_filters.get('room_type'):
        if row.get('room_type') in active_filters['room_type']:
            met.append(f"room type: {row.get('room_type')}")
        else:
            unmet.append(f"room type: {', '.join(active_filters['room_type'])}")
    
    if active_filters.get('neighbourhood'):
        hood = str(row.get('neighbourhood_group_cleansed', ''))
        if any(n.lower() in hood.lower() for n in active_filters['neighbourhood']):
            met.append(f"neighbourhood: {hood}")
        else:
            # Frame as "nearby" rather than a hard mismatch —
            # neighbourhood boundaries are fuzzy and adjacent areas often serve the same purpose
            unmet.append(f"exact neighbourhood ({', '.join(active_filters['neighbourhood'])}), but nearby")
    
    if active_filters.get('superhost'):
        if row.get('host_is_superhost') == 't':
            met.append("superhost")
        else:
            unmet.append("superhost")

    # Price is only surfaced in the reason when the user is clearly budget-conscious:
    # either they set a tight cap (≤$150) or used explicit budget language in their query.
    # Mentioning price otherwise makes the reason feel transactional and irrelevant.
    price_is_relevant = (
        active_filters.get('max_price') and active_filters['max_price'] <= 150
    ) or any(w in user_input.lower() for w in ['cheap', 'budget', 'affordable', '저렴', '싼'])
    
    if price_is_relevant and active_filters.get('max_price'):
        if row.get('price_clean', 0) <= active_filters['max_price']:
            met.append(f"under ${active_filters['max_price']}")
        else:
            unmet.append(f"under ${active_filters['max_price']}")

    met_str = ', '.join(met) if met else 'none'
    unmet_str = ', '.join(unmet) if unmet else 'none'

    prompt = f"""
You are a knowledgeable Seattle local. Write 2 sentences explaining why this listing suits the user.

Priority order when writing your reason:
1. PRICE — only mention if the user set a low price filter (≤$150) or used words like "cheap/budget/affordable". Otherwise NEVER mention price.
2. LOCATION — if neighbourhood doesn't match exactly, say it's close to or convenient for [matched area]. Never say "doesn't match" — frame it positively as a nearby alternative.
3. OTHER requests — only address conditions the user explicitly asked for. NEVER invent or mention conditions the user didn't request.

User's request: "{user_input}"
Conditions met: {met_str}
Conditions NOT met: {unmet_str}

Listing:
- Name: {row.get('name', '')}
- Neighbourhood: {row.get('neighbourhood_cleansed', '')}
- Price: {row.get('price', '')}
- Rating: {row.get('review_scores_rating', '')}
- Description: {str(row.get('description', ''))[:300]}
- Reviews: {str(row.get('review_text', ''))[:500]}

Rules:
- If unmet conditions exist, acknowledge naturally: "While it's not [X], it excels at [strength]."
- If NO active filters at all: focus purely on what the listing offers that matches the user's text query.
- NEVER mention conditions the user didn't set. If user didn't ask about parking, don't mention parking.
- For location, NEVER just say "not in [neighbourhood]" — instead say "just [X] minutes from [neighbourhood]" if reviews/description support it, or simply highlight the listing's own area strengths.
- Do not say "based on the description" or "according to reviews".
- 2 sentences max.
"""
    # Temperature 0.7: enough creativity to sound natural,
    # low enough to stay grounded in the listing data
    response = client.chat.completions.create(
        model=""gpt-4o-mini"",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# ── UI Layout ─────────────────────────────────────────────────
st.markdown('<div class="hero-title">Find your<br><i>perfect stay</i> in Seattle.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered search across thousands of listings</div>', unsafe_allow_html=True)

search_input = st.text_input(
    label="search",
    placeholder='e.g. "Cozy home near Capitol Hill, close to coffee shops"',
    label_visibility="collapsed"
)

col1, col2, col3, col4, col5 = st.columns([2.5, 2.5, 2, 2, 1.5])

with col1:
    room_types = st.multiselect(
        "Room Type",
        options=["Entire home/apt", "Private room", "Shared room"],
        default=[]
    )

with col2:
    neighbourhood_groups = sorted(listings['neighbourhood_group_cleansed'].dropna().unique().tolist())
    selected_neighbourhoods = st.multiselect(
        "Neighbourhood",
        options=neighbourhood_groups,
        default=[]
    )

with col3:
    st.markdown("**Price / night**")
    min_price, max_price = st.slider("Price", 0, 1000, (0, 300), step=10, format="$%d", label_visibility="collapsed")

with col4:
    rating_options = {"Any": 0.0, "4.0⭐ +": 4.0, "4.5⭐ +": 4.5, "4.8⭐ +": 4.8, "5.0⭐ only": 5.0}
    min_rating_label = st.selectbox("Min Rating", list(rating_options.keys()))
    min_rating = rating_options[min_rating_label]

with col5:
    superhost = st.checkbox("Superhost only", value=False)

top_k = 10
search_clicked = st.button("Search →")

# ── Search Execution ──────────────────────────────────────────
price_filter_active = (min_price > 0 or max_price < 1000)

if search_clicked and search_input.strip():
    matched_filters = {
        'room_type': room_types if room_types else None,
        'superhost': superhost if superhost else None,
        'min_rating': min_rating if min_rating > 0 else None,
        'max_price': max_price if price_filter_active else None,
        'min_price': min_price if price_filter_active else None,
        'min_rating': min_rating if min_rating > 0 else None,
        'neighbourhood': selected_neighbourhoods if selected_neighbourhoods else None,
    }

    with st.spinner("Finding your perfect stay..."):
        results = search_listings(
            search_input,
            room_type=matched_filters['room_type'],
            superhost=matched_filters['superhost'],
            max_price=matched_filters['max_price'],
            neighbourhood=matched_filters['neighbourhood'],
            top_k=top_k
        )

    st.markdown(f'<div class="result-count">Found {len(results)} listings for you.</div>', unsafe_allow_html=True)

    cards_col, map_col = st.columns([1.2, 1])

    with map_col:
        map_data = results[['latitude', 'longitude', 'name']].dropna()
        st.markdown("#### &nbsp;")
        st.map(map_data, zoom=11, use_container_width=True, height=700)

    with cards_col:
        st.markdown("#### Top Results")
        with st.container(height=700):
            for i, (_, row) in enumerate(results.iterrows()):
                with st.container(border=True):
                    img_url = row.get('picture_url', '')
                    if img_url:
                        st.image(img_url, use_container_width=True)

                    sim = row.get('similarity_score', 0)
                    is_superhost = row.get('host_is_superhost') == 't'

                    st.markdown(f"**{row.get('name', '')}** &nbsp; `{sim:.0%} match`")
                    st.caption(f"📍 {row.get('neighbourhood_cleansed', '')} · {row.get('room_type', '')} · 🛏 {row.get('beds', '')} beds · {row.get('bathrooms_text', '')}")

                    cols = st.columns([1, 1, 1])
                    cols[0].markdown(f"**{row.get('price', '')}**")
                    cols[1].markdown(f"⭐ {row.get('review_scores_rating', '')}")
                    if is_superhost:
                        cols[2].markdown("🏆 Superhost")

                    if i < 3:
                        with st.spinner("Generating reason..."):
                            try:
                                time.sleep(2)
                                reason = generate_reason(search_input, row, matched_filters)
                                reason = reason.encode('utf-8', errors='ignore').decode('utf-8')
                                reason = reason.replace('$', '\\$')
                                st.info(f"💬 {reason}")
                            except Exception as e:
                                st.warning(f"⚠️ {str(e)}")

                    st.markdown(f"[View on Airbnb →]({row.get('listing_url', '#')})")

elif search_clicked and not search_input.strip():
    st.info("Please enter a search query.")
