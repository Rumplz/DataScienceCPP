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
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS (The "Premium" Look)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Karachi Estate AI",
    page_icon="nt",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Metrics Styling */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #1abc9c;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eaeaea;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING (Cached for Speed)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("Karachi_Property_Dataset.csv")
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'Karachi_Property_Dataset.csv' is in the directory.")
        return None, None

    # Helper function for price conversion
    def convert_price(x):
        if pd.isna(x) or str(x).strip().lower() in ["na", "nan", "none", ""]:
            return np.nan
        x = str(x).replace("PKR", "").strip()
        if "Crore" in x:
            return float(x.split()[0]) * 10_000_000
        elif "Lakh" in x:
            return float(x.split()[0]) * 100_000
        else:
            try:
                return float(x.split()[0])
            except:
                return np.nan

    # Apply transformations
    df["Price"] = df["Price"].apply(convert_price)
    df["Size"] = pd.to_numeric(df["Size/Area"].str.split(" ", expand=True)[0], errors="coerce")
    df["Bedrooms"] = pd.to_numeric(df["Bedrooms"], errors="coerce")
    df["Bathrooms"] = pd.to_numeric(df["Bathrooms"], errors="coerce")
    
    # Neighborhood extraction
    temp_df = df["Address"].str.split(",", expand=True)
    df["Neighborhood"] = temp_df[0].str.strip() + " " + temp_df[1].str.strip().fillna("")
    
    # Cleaning
    df.dropna(subset=["Price", "Size", "Bedrooms", "Neighborhood"], inplace=True)
    df = df[df["Price"] > 100000] 
    df = df[df["Size"] > 10]
    
    # Outlier Removal (10th-90th percentile)
    low, high = df["Price"].quantile([0.1, 0.9])
    df = df[df["Price"].between(low, high)]
    
    return df

df = load_and_clean_data()

# -----------------------------------------------------------------------------
# 3. MODEL TRAINING (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_model(df):
    features = ["Size", "Neighborhood", "Bedrooms", "Bathrooms"]
    target = "Price"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(
        OneHotEncoder(use_cat_names=True, handle_unknown="ignore"), 
        SimpleImputer(strategy="mean"),
        StandardScaler(), 
        Ridge()
    )
    
    model.fit(X_train, y_train)
    return model, features

if df is not None:
    model, model_features = train_model(df)
else:
    st.stop()

# -----------------------------------------------------------------------------
# 4. SIDEBAR - USER INPUTS
# -----------------------------------------------------------------------------
st.sidebar.header("🏡 Property Details")
st.sidebar.markdown("Configure the property parameters below:")

with st.sidebar.form("prediction_form"):
    # Neighborhood Dropdown (Sorted alphabetically)
    neighborhood_list = sorted(df["Neighborhood"].unique().tolist())
    neighborhood = st.selectbox("Select Neighborhood", options=neighborhood_list)
    
    # Numeric Inputs
    area_size = st.number_input("Area Size (Sq. Yards)", min_value=50, max_value=5000, value=120, step=10)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
    
    submit_button = st.form_submit_button("💰 Predict Price")

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.title("Karachi Real Estate AI Validator")
st.markdown("### Intelligent Property Valuation Engine")
st.markdown("---")

if submit_button:
    # Prepare Input Data
    input_data = pd.DataFrame({
        "Size": [area_size],
        "Neighborhood": [neighborhood],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms]
    })

    # Predict
    with st.spinner("Calculating market valuation..."):
        prediction = model.predict(input_data)[0]

    # --- Formatting the Result ---
    # Helper to format large numbers nicely
    def format_crore_lakh(amount):
        if amount >= 10_000_000:
            val = amount / 10_000_000
            return f"{val:.2f} Crore"
        elif amount >= 100_000:
            val = amount / 100_000
            return f"{val:.2f} Lakh"
        else:
            return f"{amount:,.0f} PKR"

    formatted_price = format_crore_lakh(prediction)
    
    # Calculate comparisons
    avg_neighbor_price = df[df['Neighborhood'] == neighborhood]['Price'].mean()
    price_diff = prediction - avg_neighbor_price
    diff_percent = (price_diff / avg_neighbor_price) * 100

    # --- UI Layout for Results ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Estimated Value")
        st.metric(label="Predicted Market Price", value=formatted_price)
        
        st.markdown(f"""
        <div style='background-color: #f1f3f4; padding: 15px; border-radius: 5px; font-size: 0.9em;'>
            <strong>Quick Stats for {neighborhood}:</strong><br>
            • Avg Price: {format_crore_lakh(avg_neighbor_price)}<br>
            • This property is <span style='color: {"red" if diff_percent > 0 else "green"};'>
            {abs(diff_percent):.1f}% {'Higher' if diff_percent > 0 else 'Lower'}</span> than the area average.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Plotly Chart: Distribution of prices in that neighborhood
        st.subheader("Neighborhood Price Distribution")
        
        neighbor_data = df[df['Neighborhood'] == neighborhood]
        
        fig = px.histogram(
            neighbor_data, 
            x="Price", 
            nbins=20, 
            title=f"Price Trends in {neighborhood}",
            color_discrete_sequence=['#2c3e50']
        )
        
        # Add a vertical line for the predicted price
        fig.add_vline(x=prediction, line_dash="dash", line_color="#e74c3c", annotation_text="Your Prediction")
        
        # Clean up the chart look
        fig.update_layout(
            xaxis_title="Price (PKR)",
            yaxis_title="Count of Properties",
            showlegend=False,
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Feature Importance Section ---
    st.markdown("---")
    st.subheader("What is driving this price?")
    
    # Extract coefficients
    coefficients = model.named_steps['ridge'].coef_
    features_names = model.named_steps['onehotencoder'].get_feature_names_out()
    
    # Create a nice dataframe for plotting
    feat_df = pd.DataFrame({'Feature': features_names, 'Impact': coefficients})
    
    # Filter for relevant features (Size, Rooms, and specific Neighborhood impact)
    # We highlight the impact of the selected neighborhood specifically
    relevant_feats = feat_df[
        feat_df['Feature'].isin(['Size', 'Bedrooms', 'Bathrooms', f'Neighborhood_{neighborhood}'])
    ].sort_values(by="Impact", ascending=True)

    fig_imp = px.bar(
        relevant_feats, 
        x='Impact', 
        y='Feature', 
        orientation='h',
        title="Impact of Key Features on Price",
        color='Impact',
        color_continuous_scale='Teal'
    )
    fig_imp.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_imp, use_container_width=True)

else:
    # Default landing state
    st.info("👈 Please enter the property details in the sidebar and click 'Predict Price' to begin.")
    
    # Show a general dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Properties Analyzed", f"{len(df):,}")
    col2.metric("Average Market Price", format_crore_lakh(df["Price"].mean()))
    col3.metric("Locations Covered", len(df["Neighborhood"].unique()))