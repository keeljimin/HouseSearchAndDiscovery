import streamlit as st
import pandas as pd
import numpy as np
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
    from groq import Groq
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    return client, model

@st.cache_data
def load_data():
    file_id = "1eNPJJztqgoPc1-yAJoD34ua1RoHR3gX7"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    listings = pd.read_csv(url)
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
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
    
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
    matched = []
    unmatched = []

    if matched_filters.get('room_type'):
        room_type_str = ', '.join(matched_filters['room_type'])
        if row.get('room_type') in matched_filters['room_type']:
            matched.append(room_type_str)
        else:
            unmatched.append(room_type_str)

    if matched_filters.get('superhost'):
        if row.get('host_is_superhost') == 't':
            matched.append("superhost")
        else:
            unmatched.append("superhost")

    if matched_filters.get('max_price'):
        if row.get('price_clean', 0) <= matched_filters['max_price']:
            matched.append(f"under ${matched_filters['max_price']}")
        else:
            unmatched.append(f"under ${matched_filters['max_price']}")

    if matched_filters.get('neighbourhood'):
        if any(n.lower() in str(row.get('neighbourhood_group_cleansed', '')).lower() for n in matched_filters['neighbourhood']):
            matched.append(f"near {', '.join(matched_filters['neighbourhood'])}")
        else:
            unmatched.append(f"near {', '.join(matched_filters['neighbourhood'])}")

    matched_str = ', '.join(matched) if matched else 'none'
    unmatched_str = ', '.join(unmatched) if unmatched else 'none'

    desc = row.get('description')
    desc_text = str(desc)[:300] if pd.notna(desc) else 'No description available'
    review = row.get('review_text')
    review_text = str(review)[:500] if pd.notna(review) else ''

    prompt = f"""
You are a knowledgeable Seattle local who knows every neighborhood intimately. 
Never say things like "based on the description" or "according to reviews" or "the listing mentions".
Explain why this listing is a good match in 2 sentences max. Be specific and friendly.

User's request: "{user_input}"
Matched conditions: {matched_str}
Unmet conditions: {unmatched_str}

Listing:
- Name: {row.get('name', '')}
- Neighbourhood: {row.get('neighbourhood_cleansed', '')}
- Price: {row.get('price', '')}
- Rating: {row.get('review_scores_rating', '')}
- Description: {desc_text}
- Recent reviews: {review_text}

Rules:
- If some conditions are matched, mention them specifically.
- If some conditions are NOT met, acknowledge it naturally: "Although it's not [unmet condition], it stands out for [strength]."
- If NO conditions are matched at all, start with: "There are no listings in Seattle that match all your criteria, but this one stands out for [strength]."
- Only mention price if the user's max price filter is $150 or under (user's max price: {matched_filters.get('max_price', 'not set')}) OR if the user explicitly used words like "cheap", "budget", "affordable". Otherwise, NEVER mention price.
- Reference what guests actually said if relevant.
- If the user mentions transit or safety, do NOT just rely on what the listing claims. Only mention it if multiple reviews confirm it.
- If the user mentioned a landmark, restaurant, grocery store, school, or office, check ONLY the listing description and reviews provided. If found word-for-word, mention it by name or quote the review. If not found, do not mention it at all. NEVER invent or assume nearby places from your own knowledge.
- 2 sentences max.
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
if search_clicked and search_input.strip():
    matched_filters = {
        'room_type': room_types if room_types else None,
        'superhost': superhost if superhost else None,
        'min_rating': min_rating if min_rating > 0 else None,
        'max_price': max_price if max_price < 1000 else None,
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
                                time.sleep(1)
                                reason = generate_reason(search_input, row, matched_filters)
                                reason = reason.encode('utf-8', errors='ignore').decode('utf-8')
                                reason = reason.replace('$', '\\$')
                                st.info(f"💬 {reason}")
                            except Exception:
                                st.warning(f"⚠️ {str(e)}")

                    st.markdown(f"[View on Airbnb →]({row.get('listing_url', '#')})")

elif search_clicked and not search_input.strip():
    st.info("Please enter a search query.")
