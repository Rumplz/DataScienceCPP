import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Karachi Estate AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    div.stButton > button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #1abc9c;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS (MOVED TO TOP TO FIX ERROR)
# -----------------------------------------------------------------------------
def format_crore_lakh(amount):
    if amount >= 10_000_000:
        val = amount / 10_000_000
        return f"{val:.2f} Crore"
    elif amount >= 100_000:
        val = amount / 100_000
        return f"{val:.2f} Lakh"
    else:
        return f"{amount:,.0f} PKR"

# -----------------------------------------------------------------------------
# 3. DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("Karachi_Property_Dataset.csv")
    except FileNotFoundError:
        return None

    def convert_price(x):
        if pd.isna(x) or str(x).strip().lower() in ["na", "nan", "none", ""]:
            return np.nan
        x = str(x).replace("PKR", "").strip()
        if "Crore" in x:
            return float(x.split()[0]) * 10_000_000
        elif "Lakh" in x:
            return float(x.split()[0]) * 100_000
        else:
            try: return float(x.split()[0])
            except: return np.nan

    df["Price"] = df["Price"].apply(convert_price)
    df["Size"] = pd.to_numeric(df["Size/Area"].str.split(" ", expand=True)[0], errors="coerce")
    df["Bedrooms"] = pd.to_numeric(df["Bedrooms"], errors="coerce")
    df["Bathrooms"] = pd.to_numeric(df["Bathrooms"], errors="coerce")
    
    temp_df = df["Address"].str.split(",", expand=True)
    df["Neighborhood"] = temp_df[0].str.strip() + " " + temp_df[1].str.strip().fillna("")
    
    df.dropna(subset=["Price", "Size", "Bedrooms", "Neighborhood"], inplace=True)
    df = df[df["Price"] > 100000] 
    df = df[df["Size"] > 10]
    
    low, high = df["Price"].quantile([0.1, 0.9])
    df = df[df["Price"].between(low, high)]
    
    return df

df = load_and_clean_data()

if df is None:
    st.error("Dataset not found! Please upload 'Karachi_Property_Dataset.csv'")
    st.stop()

# -----------------------------------------------------------------------------
# 4. MODEL TRAINING
# -----------------------------------------------------------------------------
@st.cache_resource
def train_model(df):
    features = ["Size", "Neighborhood", "Bedrooms", "Bathrooms"]
    X = df[features]
    y = df["Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(
        OneHotEncoder(use_cat_names=True, handle_unknown="ignore"), 
        SimpleImputer(strategy="mean"),
        StandardScaler(), 
        Ridge()
    )
    model.fit(X_train, y_train)
    return model

model = train_model(df)

# -----------------------------------------------------------------------------
# 5. SIDEBAR & MAIN APP
# -----------------------------------------------------------------------------
st.sidebar.header("Property Details")
with st.sidebar.form("prediction_form"):
    neighborhood_list = sorted(df["Neighborhood"].unique().tolist())
    neighborhood = st.selectbox("Select Neighborhood", options=neighborhood_list)
    area_size = st.number_input("Area Size (Sq. Yards)", min_value=50, max_value=5000, value=120, step=10)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
    submit_button = st.form_submit_button("Predict Price")

st.title("Karachi Real Estate AI Validator")
st.markdown("### Intelligent Property Valuation Engine")
st.markdown("---")

if submit_button:
    # PREDICTION LOGIC
    input_data = pd.DataFrame({
        "Size": [area_size],
        "Neighborhood": [neighborhood],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms]
    })

    with st.spinner("Calculating..."):
        prediction = model.predict(input_data)[0]

    formatted_price = format_crore_lakh(prediction)
    avg_neighbor_price = df[df['Neighborhood'] == neighborhood]['Price'].mean()
    diff_percent = ((prediction - avg_neighbor_price) / avg_neighbor_price) * 100
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Estimated Value")
        st.metric(label="Predicted Price", value=formatted_price)
        status_color = "red" if diff_percent > 0 else "green"
        status_text = "Higher" if diff_percent > 0 else "Lower"
        st.markdown(f"""
        <div style='background-color: #f1f3f4; padding: 15px; border-radius: 5px; font-size: 0.9em;'>
            <strong>Stats for {neighborhood}:</strong><br>
            • Avg: {format_crore_lakh(avg_neighbor_price)}<br>
            • <span style='color: {status_color};'>{abs(diff_percent):.1f}% {status_text}</span> than average.
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        neighbor_data = df[df['Neighborhood'] == neighborhood]
        fig = px.histogram(neighbor_data, x="Price", nbins=20, title=f"Price Trends in {neighborhood}", color_discrete_sequence=['#2c3e50'])
        fig.add_vline(x=prediction, line_dash="dash", line_color="#e74c3c")
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    st.subheader("What drives this price?")
    coefs = model.named_steps['ridge'].coef_
    names = model.named_steps['onehotencoder'].get_feature_names_out()
    feat_df = pd.DataFrame({'Feature': names, 'Impact': coefs})
    relevant = feat_df[feat_df['Feature'].isin(['Size', 'Bedrooms', 'Bathrooms', f'Neighborhood_{neighborhood}'])].sort_values(by="Impact")
    fig_imp = px.bar(relevant, x='Impact', y='Feature', orientation='h', title="Feature Impact", color='Impact', color_continuous_scale='Teal')
    st.plotly_chart(fig_imp, use_container_width=True)

else:
    # LANDING PAGE LOGIC
    st.info("Please enter the property details in the sidebar.")
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Properties", f"{len(df):,}")
    
    # This line works now because 'format_crore_lakh' is defined at the top of the file
    col2.metric("Average Price", format_crore_lakh(df["Price"].mean()))
    
    col3.metric("Locations", len(df["Neighborhood"].unique()))
