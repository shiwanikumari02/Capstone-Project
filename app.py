import streamlit as st
import pandas as pd
import numpy as np
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium
import joblib
import plotly.graph_objects as go
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import re
import base64

# --------------- Streamlit Config --------------- #
st.set_page_config(page_title="GreenRoute: CO2 Optimization", page_icon="icon.png", layout="wide")

# --------------- Set Background Image --------------- #
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    .distance-box {{
        background-color: #E6FAF1;
        color: #2E8B57;
        padding: 12px;
        font-weight: bold;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
        font-size: 16px;
    }}
    h1, h2, h3, h4 {{
        color: #2E8B57;
    }}
    </style>
    """, unsafe_allow_html=True)

set_background("background_img.jpg")

# --------------- Header with Icon --------------- #
st.markdown("""
<div style='display:flex; align-items:center; gap: 1rem;'>
    <img src='data:image/png;base64,{}' width='55'/>
    <h1>GreenRoute: Carbon Footprint Optimization</h1>
</div>
""".format(base64.b64encode(open("icon.png", "rb").read()).decode()), unsafe_allow_html=True)

# --------------- Helper Functions --------------- #
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# --------------- Chatbot Section --------------- #
faq_df = pd.read_csv("co2_faqs_extended.csv")
faq_df.columns = ['question', 'answer']

def chatbot_reply(user_input: str) -> str:
    matches = faq_df[faq_df['question'].str.lower().apply(lambda q: any(w in q for w in user_input.lower().split()))]
    return matches.iloc[0]['answer'] if not matches.empty else "💭 Sorry, try rephrasing your question."

st.markdown("### 🤖 Chat about CO2 and Emissions")
user_input = st.text_input("Ask your CO₂ emission or sustainability question...")
if user_input:
    st.info(chatbot_reply(user_input))

# --------------- Load Models --------------- #
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_order = joblib.load("feature_order.pkl")
geolocator = Nominatim(user_agent='greenroute_app')

# --------------- User Inputs --------------- #
col1, col2 = st.columns(2)
with col1:
    origin = st.text_input("📍 Origin City", "Delhi")
    vehicle_type = st.selectbox("🚛 Vehicle Type", list(label_encoders['Vehicle Type'].classes_))
    fuel_type = st.selectbox("⛽ Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'])
with col2:
    destination = st.text_input("🏁 Destination City", "Mumbai")
    road_type = st.selectbox("🛣 Road Type", list(label_encoders['Road Type'].classes_))
    traffic_condition = st.selectbox("🚦 Traffic Condition", list(label_encoders['Traffic Conditions'].classes_))

# --------------- Location and Distance --------------- #
try:
    origin_loc = geolocator.geocode(origin, timeout=5)
    dest_loc = geolocator.geocode(destination, timeout=5)
except:
    st.error("❌ Location lookup failed. Please check your input.")
    st.stop()

if origin_loc and dest_loc:
    origin_coords = (origin_loc.latitude, origin_loc.longitude)
    dest_coords = (dest_loc.latitude, dest_loc.longitude)
    distance_km = geodesic(origin_coords, dest_coords).kilometers

    st.markdown(f"<div class='distance-box'>📏 Distance: {distance_km:.1f} km</div>", unsafe_allow_html=True)

    # --------------- Predict CO2 --------------- #
    input_df = pd.DataFrame([{col: 0 for col in feature_order}])
    input_df.at[0, 'Distance(Km)'] = distance_km
    input_df.at[0, 'Engine Size'] = 2.0
    input_df.at[0, 'Age of Vehicle'] = 3
    input_df.at[0, 'Mileage'] = 10000
    input_df.at[0, 'Speed'] = 60
    input_df.at[0, 'Acceleration'] = 2.5
    input_df.at[0, 'Temperature'] = 25
    input_df.at[0, 'Humidity'] = 50
    input_df.at[0, 'Wind Speed'] = 5
    input_df.at[0, 'Air Pressure'] = 1010
    input_df.at[0, 'Vehicle Type'] = label_encoders['Vehicle Type'].transform([vehicle_type])[0]
    input_df.at[0, 'Road Type'] = label_encoders['Road Type'].transform([road_type])[0]
    input_df.at[0, 'Traffic Conditions'] = label_encoders['Traffic Conditions'].transform([traffic_condition])[0]
    input_df.at[0, 'Fuel Type'] = {'Petrol': 1, 'Diesel': 2, 'CNG': 3, 'Electric': 4}[fuel_type]

    co2_per_km = model.predict(input_df)[0]
    co2_emission = co2_per_km * distance_km

    # --------------- Sidebar Tips --------------- #
    with st.sidebar.expander("💡 Eco Tips"):
        st.markdown("""
        - 📦 Optimize trips and cargo loads
        - 🔄 Avoid empty returns
        - 🚀 Drive efficiently and avoid idling
        - ⚡ Switch to electric fleets
        - 🌍 Plan eco-friendly routes
        """)

    # --------------- Sidebar Simulator --------------- #
    st.sidebar.markdown("### 🧪 CO₂ Emission Simulator")
    sim_speed = st.sidebar.slider("🚗 Simulated Speed", 30, 120, 60)
    sim_fuel = st.sidebar.selectbox("⛽ Simulated Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'])
    sim_df = input_df.copy()
    sim_df.at[0, 'Speed'] = sim_speed
    sim_df.at[0, 'Fuel Type'] = {'Petrol': 1, 'Diesel': 2, 'CNG': 3, 'Electric': 4}[sim_fuel]
    sim_emission = model.predict(sim_df)[0] * distance_km
    st.sidebar.info(f"📉 Emission at {sim_speed} km/h on {sim_fuel}: **{sim_emission:.2f} g**")

    # --------------- Gauge Chart --------------- #
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=co2_emission,
        title={'text': "CO₂ Emission (g)", 'font': {'color': "#212529"}},
        gauge={
            'axis': {'range': [0, 500]},
            'bar': {'color': "#00C853" if co2_emission < 100 else "#FFD600" if co2_emission < 200 else "#D50000"},
            'bgcolor': "white"
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # --------------- Suggestions --------------- #
    st.markdown("### 🌱 Suggestions to Reduce CO₂ Emissions")
    if co2_emission < 100:
        st.success("✅ Low Emissions")
    elif co2_emission < 200:
        st.warning("⚠️ Moderate Emissions")
    else:
        st.error("🚨 High Emissions")

    # Tips
    st.markdown("""
    - Use public transport or carpooling
    - Maintain your vehicle regularly
    - Switch to CNG or electric
    - Avoid peak traffic hours
    """)

    # --------------- Map --------------- #
    m = folium.Map(location=origin_coords, zoom_start=5)
    folium.Marker(origin_coords, popup=origin).add_to(m)
    folium.Marker(dest_coords, popup=destination).add_to(m)
    folium.PolyLine([origin_coords, dest_coords], color='green').add_to(m)
    st_folium(m, width=950)

    # --------------- Country Emissions --------------- #
    emissions_df = pd.read_csv("country_emissions.csv")
    emissions_df.columns = emissions_df.columns.str.lower().str.strip()

    country_name = st.selectbox("📊 Country Analysis", sorted(emissions_df["country"].unique()))
    country_data = emissions_df[emissions_df["country"] == country_name]
    if not country_data.empty:
        st.success(f"{country_name} emitted {country_data['co2'].sum():.2f} Mt CO₂.")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.lineplot(data=country_data, x="year", y="co2", ax=ax[0])
        ax[0].set_title("Total CO₂")
        sns.lineplot(data=country_data, x="year", y="co2_per_capita", ax=ax[1])
        ax[1].set_title("Per Capita CO₂")
        st.pyplot(fig)

    # --------------- PDF Download --------------- #
   # --------------- PDF Download with Suggestions --------------- #
if st.button("📄 Download CO2 Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="GreenRoute CO2 Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Origin: {remove_emojis(origin)}", ln=True)
    pdf.cell(200, 10, txt=f"Destination: {remove_emojis(destination)}", ln=True)
    pdf.cell(200, 10, txt=f"Distance: {distance_km:.1f} km", ln=True)
    pdf.cell(200, 10, txt=f"Estimated CO2: {co2_emission:.2f} g", ln=True)

    # Add a blank line before suggestions
    pdf.cell(200, 10, txt="", ln=True)

    # Suggestions Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Suggestions to Reduce CO2 Emissions:", ln=True)
    pdf.set_font("Arial", size=11)

    suggestions = ""
    if co2_emission < 100:
        suggestions += "- Use public transport or carpooling\n- Maintain your vehicle regularly\n"
    elif 100 <= co2_emission < 200:
        suggestions += "- Switch to CNG or Electric vehicle\n- Avoid peak traffic\n- Use eco-driving habits\n"
    else:
        suggestions += "- Shift to Electric/Hybrid vehicle\n- Optimize cargo and routes\n- Avoid heavy vehicles for light loads\n"

    if fuel_type in ['Petrol', 'Diesel']:
        suggestions += f"- Consider switching from {fuel_type} to Electric\n"
    if road_type == 'Urban':
        suggestions += "- Urban roads increase idling. Try expressways or alternate routes\n"
    if traffic_condition == 'Heavy':
        suggestions += "- Heavy traffic increases emissions. Shift travel time if possible\n"
    if vehicle_type in ['Heavy Truck', 'Bus']:
        suggestions += f"- Using {vehicle_type}? Avoid empty returns and optimize cargo\n"

    pdf.multi_cell(0, 10, txt=remove_emojis(suggestions))

    # Save and trigger download
    pdf.output("GreenRoute_CO2_Report.pdf")
    st.download_button("⬇️ Download PDF", data=open("GreenRoute_CO2_Report.pdf", "rb").read(),
                       file_name="GreenRoute_CO2_Report.pdf", mime="application/pdf")

