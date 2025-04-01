import streamlit as st
import pandas as pd
import os
from langfuse.decorators import observe
from langfuse.openai import OpenAI
import boto3
from dotenv import load_dotenv
from pycaret.regression  import load_model, predict_model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#wczytanie zmiennych 
load_dotenv()
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#wczytanie modelu
MODEL_NAME = 'huber_model_halfmarathon_time'
model = load_model(MODEL_NAME)


# Strona główna aplikacji
st.set_page_config(layout="centered")
st.title(' 🏅 Biegowy Prognozator 🏃')
st.write("---")
st.markdown("Chcesz wiedzieć, ile zajmie Ci ukończenie półmaratonu? 🏃‍♂️" 
"Wprowadź swoje tempo i czas na 5 km, a my oszacujemy Twój wynik na mecie!" 
 "Dodatkowo otrzymasz krótką poradę treningową od AI oraz zobaczysz, jak wypadasz na tle uczestników półmaratonu wrocławskiego!")
                   
tab1, tab2, tab3 = st.tabs(["Poznaj swój czas ⏳", "Krótka porada od AI 💡", "Ty kontra inni zawodnicy 📊"])

# Funkcja do konwersji sekund na h:m:s
def convert_seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


# wygenerowanie porady od AI
@observe()
def get_training_advice(wiek, płeć, tempo, czas):    
    prompt = f"Użytkownik ma {wiek} lat, płeć: {płeć}, tempo na 5 km: {tempo} min/km, czas na 5 km: {czas} minut. Podaj krótką, zwięzłą i bardzo konkretną poradę dotyczącą treningu tak by zawodnik uzyskał jeszcze lepszy czas półmaratonu,bez urywania zdań. "
    response = openai_client.chat.completions.create(
        model="gpt-4o", # model AI
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0
    )
    return response.choices[0].message.content.strip()



#zawartość tab 1, wartości, na których trenowany był model
with tab1:
    st.subheader("Sprawdź Twój estymowany czas półmaratonu")
    st.caption("Wypełnij wszystkie pola poniżej i naciśnij przycisk.")

    płeć=st.radio("Wybierz płeć 👇", 
              ['Kobieta', 'Mężczyzna']
    )
    wiek = st.slider("Ile masz lat 👇", 0, 100, 50)
    tempo = st.number_input("Wpisz swoje tempo na 5 km (w minutach na 5 km, np.6.45) 👇",  min_value=0.0, format="%.2f",  
    )
    czas = st.number_input("Wpisz swój czas na 5 km (w minutach, np. 35) 👇", value=0, step=1, min_value=0)
    
    
    # przycisk generujący przewidywany czas
    # Tworzymy DataFrame do predykcji
    if st.button("Pokaż mój czas"):
        if tempo <= 0 or czas <= 0: #pola muszą być wypełnione i większe niż 0
           st.error("Proszę wprowadzić prawidłowe wartości dla tempa i czasu!")
        else:
            czas_sekundy = czas * 60 #zmiana na sekundy jak w modelu
        # Przygotowanie danych w odpowiednim formacie, takim na jakim był trenowany model
            dane_użytkownika = pd.DataFrame({
                'Wiek': [wiek],
                'Płeć': [płeć],
                '5 km Tempo': [tempo],
                '5 km Czas': [czas_sekundy]
            })

            # Konwersja płci na liczby
            dane_użytkownika['Płeć'] = dane_użytkownika['Płeć'].map({'Kobieta': 0, 'Mężczyzna': 1})
            #Przewidywanie czasu
            with st.spinner('Przewidywanie czasu...'):
                predykcja = predict_model(model, data=dane_użytkownika)
                czas_sekundy = predykcja["prediction_label"][0]  # Czas w sekundach
                # Konwersja na format HH:MM:SS
                czas_format = convert_seconds_to_hms(czas_sekundy)

            # Zapisanie wyniku w sesji
            st.session_state.czas_format = czas_format
            st.session_state.tempo = tempo
            st.session_state.czas = czas
            st.session_state.wiek = wiek
            st.session_state.płeć = płeć

            st.success(f"Twój przewidywany czas na półmaraton to: {czas_format} ⏱️")
            # Generowanie porady od AI i zapisywanie jej w session_state
            porada = get_training_advice(wiek, płeć, tempo, czas)
            st.session_state.porada = porada

            
    

# Tab2: Porada treningowa od AI
with tab2:
    st.subheader("Treningowa Porada AI")
    st.caption("AI może pomóc w treningu, ale nie zastępuje doświadczenia i wiedzy trenera. Zawsze konsultuj się z profesjonalistą. ")
    if 'porada' in st.session_state:
        st.write("Porada treningowa od AI: ", st.session_state.porada)
    else:
        st.write("Porada nie została jeszcze wygenerowana. Wróć do zakładki 'Poznaj swój czas' i oblicz czas.")



# Tab3: klika wykresów dla czasu i tempa na 5 km 
with tab3:
    st.subheader("Wizualizacja wyników")
    st.caption("Sprawdź, jak Twój czas i tempo na 5 km plasują się do wyników uczestników półmaratonu wrocławskiego z 2023 i 2024 roku.")
    
    #sprawdzam czy użytkownik wpisał dane
    if 'czas' not in st.session_state or 'tempo' not in st.session_state:
        st.write("Wykresy nie zostały jeszcze wygenerowane. Wróć do zakładki 'Poznaj swój czas' i oblicz czas.")
    else:
        df = pd.read_csv("df_cleaned.csv")
        # Przekształcenie kolumny '5 km Czas' z sekund na minuty
        df['5 km Czas'] = df['5 km Czas'] / 60  


        # Dodanie użytkownika do danych
        user_data = pd.DataFrame({
            'Wiek': [st.session_state.wiek],
            'Płeć': [0 if st.session_state.płeć == 'Kobieta' else 1],
            '5 km Tempo': [st.session_state.tempo],
            '5 km Czas': [st.session_state.czas]  
        })
        df = pd.concat([df, user_data], ignore_index=True)
    

        # Tworzenie wykresu, czas na 5km na tle wieku innych biegaczy - wykres 1
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df['Wiek'], df['5 km Czas'], alpha=0.5, label="Inni biegacze", color='blue')
        ax.scatter(st.session_state.wiek, st.session_state.czas, color='red', label="Twój wynik", s=100, edgecolors='black')
        
        ax.set_xlabel("Wiek")
        ax.set_ylabel("Czas na 5 km (min)")
        ax.set_title("Twój czas na 5 km na tle wieku ")
        ax.legend()
    
        st.pyplot(fig)
    
      
        # Histogram - porównanie tempa użytkownika z innymi biegaczami - wykres 2
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['5 km Tempo'], bins=20, kde=True, color='green', alpha=0.5, label="Inni biegacze")
        ax.axvline(st.session_state.tempo, color='red', linestyle='--', linewidth=2, label="Twoje tempo")
        
        ax.set_xlabel("Tempo na 5 km (min/km)")
        ax.set_ylabel("Liczba zawodników")
        ax.set_title("Rozkład tempa na 5 km wśród biegaczy")
        ax.legend()
        
        st.pyplot(fig)  

        
        # Histogram - porównanie czasu na 5km użytkownika z innymi biegaczami - wykres 3
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['5 km Czas'], bins=20, kde=True, color='purple', alpha=0.5, label="Inni biegacze")
        ax.axvline(st.session_state.czas, color='red', linestyle='--', linewidth=2, label="Twój czas")
        
        ax.set_xlabel("Czas na 5 km (min)")
        ax.set_ylabel("Liczba zawodników")
        ax.set_title("Rozkład czasu na 5 km wśród biegaczy")
        ax.legend()
        
        st.pyplot(fig)



    
