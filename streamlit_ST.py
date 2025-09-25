import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go

# Ladda modellen och lag
@st.cache_resource
def load_model():
    with open('football_value_model_ST.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['features'], model_data['r2_score']

@st.cache_data
def load_teams():
    df = pd.read_csv('data/final_data.csv')
    
    # Filter out non-European leagues (same as in training notebook)
    brazilian_teams = [
        'Grêmio FBPA', 'SC Internacional', 'Clube de Regatas do Flamengo', 'Fluminense FC',
        'Botafogo de Futebol e Regatas', 'CR Vasco da Gama', 'São Paulo FC', 'SE Palmeiras',
        'Sport Club Corinthians Paulista', 'Santos FC', 'CA Mineiro', 'Cruzeiro EC',
        'Fortaleza EC', 'Ceará SC', 'Sport Recife', 'CA Paranaense', 'Coritiba FC',
        'Goiás EC', 'CA Bragantino', 'Cuiabá EC', 'América FC (MG)', 'Juventude',
        'Chapecoense AF', 'ABC FC'
    ]
    
    mls_teams = [
        'New York City FC', 'New England Revolution', 'DC United', 'Philadelphia Union',
        'Toronto FC', 'CF Montréal', 'Orlando City SC', 'Inter Miami CF', 'New York Red Bulls',
        'Chicago Fire FC', 'FC Cincinnati', 'Columbus Crew', 'Atlanta United FC',
        'Charlotte FC', 'Nashville SC', 'Sporting Kansas City', 'Minnesota United FC',
        'FC Dallas', 'Austin FC', 'Houston Dynamo', 'Colorado Rapids', 'Real Salt Lake',
        'LA Galaxy', 'Los Angeles FC', 'San Jose Earthquakes', 'Seattle Sounders FC',
        'Portland Timbers', 'Vancouver Whitecaps'
    ]
    
    argentine_teams = [
        'River Plate', 'CA Boca Juniors', 'Racing Club', 'CA Independiente',
        'San Lorenzo de Almagro', 'Estudiantes de La Plata', 'CA Lanús',
        'Defensa y Justicia', 'CA Talleres', 'Rosario Central', 'Newell\'s Old Boys',
        'Gimnasia y Esgrima La Plata', 'CA Vélez Sarsfield', 'Arsenal de Sarandí',
        'Banfield', 'Colón de Santa Fe', 'CA Unión', 'Aldosivi', 'Patronato',
        'CA Huracán', 'Godoy Cruz', 'Platense', 'Central Córdoba', 'Sarmiento de Junín',
        'Barracas Central', 'Tigre', 'CA Belgrano'
    ]
    
    mexican_teams = [
        'Club América', 'Guadalajara', 'Cruz Azul', 'UNAM', 'Tigres UANL', 'CF Monterrey',
        'Santos Laguna', 'Club León', 'Toluca', 'Pachuca', 'Atlas', 'Puebla FC',
        'Querétaro FC', 'FC Juárez', 'Tijuana', 'Necaxa', 'Mazatlán FC'
    ]
    
    south_african_teams = [
        'Mamelodi Sundowns', 'Orlando Pirates', 'Kaizer Chiefs', 'SuperSport United FC',
        'Cape Town City FC', 'Stellenbosch FC', 'AmaZulu FC', 'Golden Arrows',
        'Sekhukhune United FC', 'Royal AM', 'Chippa United', 'Maritzburg United',
        'Baroka FC', 'TS Galaxy', 'Swallows FC'
    ]
    
    japanese_teams = [
        'Urawa Red Diamonds', 'Kashima Antlers', 'Gamba Osaka', 'Yokohama F. Marinos',
        'Kawasaki Frontale', 'Cerezo Osaka', 'Vissel Kobe', 'FC Tokyo', 'Nagoya Grampus',
        'Sanfrecce Hiroshima', 'Kashiwa Reysol', 'Sagan Tosu', 'Avispa Fukuoka',
        'Shonan Bellmare', 'Albirex Niigata', 'Hokkaido Consadole Sapporo', 'Kyoto Sanga',
        'Tokushima Vortis', 'Ventforet Kofu'
    ]
    
    saudi_teams = [
        'Al-Hilal Saudi FC', 'Al-Nassr FC', 'Al-Ahli Saudi FC', 'Al-Ittihad Club (Jeddah)',
        'Al-Shabab Club (Riyadh)', 'Al-Fateh SC', 'Al-Taawoun FC', 'Ettifaq FC',
        'Al-Raed SFC', 'Al-Fayha FC', 'Al-Wehda FC', 'Al-Batin FC', 'Al-Adalah FC',
        'Al-Khaleej', 'Al-Tai', 'Abha Club', 'Damac FC'
    ]
    
    korean_teams = [
        'Jeonbuk Hyundai Motors', 'Ulsan Hyundai', 'Pohang Steelers', 'Suwon Samsung Bluewings',
        'FC Seoul', 'Daegu FC', 'Incheon United', 'Jeju United', 'Gangwon FC',
        'Gwangju FC', 'Daejeon Hana Citizen', 'Seongnam FC', 'Suwon FC'
    ]
    
    australian_teams = [
        'Sydney FC', 'Melbourne City FC', 'Melbourne Victory', 'Western Sydney Wanderers FC',
        'Adelaide United', 'Brisbane Roar', 'Perth Glory', 'Wellington Phoenix',
        'Central Coast Mariners', 'Newcastle United Jets', 'Western United FC',
        'Macarthur FC'
    ]
    
    # Combine all teams to remove
    teams_to_remove = (
        brazilian_teams + mls_teams + argentine_teams + mexican_teams +
        south_african_teams + japanese_teams + saudi_teams + korean_teams + australian_teams
    )
    
    # Filter out non-European leagues
    df_filtered = df[~df['team'].isin(teams_to_remove)]
    teams = sorted(df_filtered['team'].unique())
    return teams



def validate_input(value, min_val, max_val, field_name):
    """Validera användarinput"""
    if value < min_val or value > max_val:
        st.warning(f"{field_name} bör vara mellan {min_val} och {max_val}")
    return value

def create_spider_chart(height, age, goals_per_min, assists_per_min, minutes_played, award):
    """Skapa spider chart för spelarens attribut"""
    
    # Normalisera värdena till 0-100 skala för bättre visualisering
    categories = ['Längd', 'Ålder (inv)', 'Mål/minut', 'Assist/minut', 'Spelminuter', 'Utmärkelser']
    
    # Beräkna normaliserade värden (0-100)
    height_norm = min(100, max(0, (height - 150) / (220 - 150) * 100))
    age_norm = max(0, 100 - (age - 16) / (45 - 16) * 100)  # Yngre = bättre
    goals_norm = min(100, goals_per_min * 10000)  # Normalisera goals per minute
    assists_norm = min(100, assists_per_min * 10000)  # Normalisera assists per minute
    minutes_norm = min(100, minutes_played / 3000 * 100)  # Normalisera mot 3000 min
    award_norm = min(100, award / 10 * 100)  # Normalisera mot 10 utmärkelser
    
    values = [height_norm, age_norm, goals_norm, assists_norm, minutes_norm, award_norm]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Spelarattribut',
        fillcolor='rgba(255, 100, 0, 0.3)',
        line=dict(color='rgba(255, 100, 0, 1)', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Spelarattribut - Spider Chart (ST Model)",
        height=500
    )
    
    return fig

def main():
    st.title("⚽ Football Player Value Predictor - ST Model")
    st.markdown("Förutsäg marknadsvärdet för fotbollsspelare baserat på prestationer och attribut sett över 2 säsonger (ST-modell)")
    
    try:
        model, features, r2_score = load_model()
        teams = load_teams()
        st.success(f"ST-modell laddad! R² score: {r2_score:.4f} ({len(teams)} lag tillgängliga)")
    except FileNotFoundError:
        st.error("ST-modell inte hittad! Kontrollera att football_value_model_ST.pkl finns")
        return
    
    st.sidebar.header("Spelarinformation")
    
    # Input fields
    height = st.sidebar.slider("Längd (cm)", 150, 220, 180)
    age = st.sidebar.slider("Ålder", 16, 45, 25)
    
    st.sidebar.subheader("Prestationer")
    appearance = st.sidebar.number_input("Antal matcher", 0, 200, 30)
    goals_total = st.sidebar.number_input("Antal mål", 0, 200, 5)
    assists_total = st.sidebar.number_input("Antal assist", 0, 100, 3)
    
    st.sidebar.subheader("Övrigt")
    minutes_played = st.sidebar.number_input("Minuter spelade", 0, 15000, 2000)
    days_injured = st.sidebar.number_input("Dagar skadad", 0, 1000, 50)
    games_injured = st.sidebar.number_input("Matcher missade pga skada", 0, 100, 5)
    award = st.sidebar.number_input("Utmärkelser", 0, 50, 2)
    
    st.sidebar.subheader("Position")
    # ST-modellen är endast för anfallare
    position_encoded = 4
    st.sidebar.write("**Position:** Anfallare (endast för ST-modell)")
    
    winger = st.sidebar.checkbox("Ytterposition")
    winger_encoded = 1 if winger else 0
    
    # Team selection
    st.sidebar.subheader("Lag")
    selected_team = st.sidebar.selectbox(
        "Välj lag",
        options=teams,
        index=0,
        help="Välj spelarens lag från listan med 374 tillgängliga lag"
    )

    # Skapa one-hot encoding för team
    team_features = []
    for feature in features:
        if feature.startswith('team_'):
            team_name = feature[5:]  # Ta bort 'team_' prefix
            if team_name == selected_team:
                team_features.append(1)
            else:
                team_features.append(0)
    
    # Beräkna per minut värden som ST-modellen använder
    goals_per_minutesplayed = goals_total / (minutes_played + 1e-8)
    assists_per_minutesplayed = assists_total / (minutes_played + 1e-8)
    days_injured_per_minutesplayed = days_injured / (minutes_played + 1e-8)
    
    # Skapa input array med alla features (enligt ST-modell struktur)
    # Ordning: height, age, appearance, minutes played, games_injured, award, position_encoded, winger, sedan team features, sedan per-minut features
    base_features = [
        height, age, appearance, minutes_played, games_injured, award,
        position_encoded, winger_encoded
    ]
    
    # Kombinera bas features med team features och sedan per-minut features
    all_features = base_features + team_features + [
        goals_per_minutesplayed, assists_per_minutesplayed, days_injured_per_minutesplayed
    ]
    input_data = np.array([all_features])
    
    # Prediction
    if st.button("Förutsäg Marknadsvärde (ST Model)", type="primary"):
        prediction = model.predict(input_data)[0]
        
        # Formatera resultat
        if prediction < 1000000:
            value_str = f"{prediction/1000:.0f}K €"
        else:
            value_str = f"{prediction/1000000:.1f}M €"
        
        st.success(f"🎯 Förutsagt marknadsvärde (ST Model): **{value_str}**")
        
        # Visa spider chart
        st.subheader("📊 Spelarattribut - Spider Chart (ST Model)")
        spider_fig = create_spider_chart(height, age, goals_per_minutesplayed, 
                                       assists_per_minutesplayed, minutes_played, award)
        st.plotly_chart(spider_fig, use_container_width=True)
        
        # Visa input sammanfattning
        with st.expander("📊 Input Sammanfattning"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Längd:** {height} cm")
                st.write(f"**Ålder:** {age} år")
                st.write(f"**Matcher:** {appearance}")
                st.write(f"**Antal mål:** {goals_total} ({goals_per_minutesplayed:.6f}/minut)")
                st.write(f"**Antal assist:** {assists_total} ({assists_per_minutesplayed:.6f}/minut)")
            
            with col2:
                st.write(f"**Position:** Anfallare")
                st.write(f"**Ytterposition:** {'Ja' if winger else 'Nej'}")
                st.write(f"**Lag:** {selected_team}")
                st.write(f"**Minuter spelade:** {minutes_played}")
                st.write(f"**Utmärkelser:** {award}")

    # Visa modellinfo
    with st.expander("ℹ️ Om ST-Modellen"):
        st.write("**Random Forest Regressor (ST Model)**")
        st.write(f"- R² Score: {r2_score:.4f}")
        st.write("- Tränad endast på anfallare från europeiska ligor")
        st.write(f"- Förutsäger marknadsvärde baserat på {len(features)} features")
        st.write("- Använder per-minut statistik istället för per-match värden")

if __name__ == "__main__":
    main()
