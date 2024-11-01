from flask import Flask, render_template, request, jsonify
import json
from transformers import AutoModelForCausalLM, AutoTokenizer 
import re 
import requests

app = Flask(__name__)

def get_species_info(lat, lng):
    # lat = 43.6532
    # lng = -79.3832
    species_api_url = f"https://api.gbif.org/v1/occurrence/search?decimalLatitude={lat}&decimalLongitude={lng}"
    response = requests.get(species_api_url)

    response.raise_for_status()
    species_data = response.json()
    species_names = [result['species'] for result in species_data['results']]  # Adjust based on actual structure
    if len(species_names) == 0:
        species_names = [
    "Dodo (Raphus cucullatus)",
    "Passenger Pigeon (Ectopistes migratorius)",
    "Great Auk (Pinguinus impennis)",
    "Heath Hen (Tympanuchus cupido cupido)",
    "Javan Tiger (Panthera tigris sondaica)",
    "Golden Toad (Incilius periglenes)",
    "Quagga (Equus quagga quagga)"
                ]
        
    print(species_names)
    return species_names

# Open-Meteo API endpoint for weather and climate data
OPEN_METEO_URL = 'https://api.open-meteo.com/v1/forecast'

def get_weather_data(lat, lon):
    """Fetch weather data from Open-Meteo API based on latitude and longitude."""
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,precipitation',
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
        'timezone': 'auto'
    }
    response = requests.get(OPEN_METEO_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def generate_climate_summary(weather_data):
    """Generate a summary based on the retrieved weather data."""
    if not weather_data:
        return "Weather data is unavailable. Please try again later."

    # Extract relevant data for summary
    daily_data = weather_data.get('daily', {})
    temperature_max = daily_data.get('temperature_2m_max', [])[0] if daily_data.get('temperature_2m_max') else None
    temperature_min = daily_data.get('temperature_2m_min', [])[0] if daily_data.get('temperature_2m_min') else None
    precipitation = daily_data.get('precipitation_sum', [0])[0] if daily_data.get('precipitation_sum') else 0

    # Create a summary
    summary = (
        f"The maximum temperature is expected to be {temperature_max}°C, "
        f"the minimum temperature is {temperature_min}°C, "
        f"and total precipitation is {precipitation} mm for the selected location."
    )

    return summary


@app.route('/set-location', methods=['POST'])
def set_location():
    data = request.get_json()
    lat = data.get('lat')
    lng = data.get('lng')
    
    print(f"Received coordinates: Latitude={lat}, Longitude={lng}")  # Check if these are not null
    
    return jsonify({"status": "success", "latitude": lat, "longitude": lng})


model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def generate_report(lat, lng):
    prompt = f"""
    Create a focused action plan for conserving the habitat at coordinates ({lat}, {lng}). Include a Habitat Assessment, Threat Analysis, measurable Conservation Goals, Actionable Strategies, Community Engagement methods, and a Budget and Resource Plan to ensure sustainable habitat conservation.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=200,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    clean_text = re.sub(re.escape(prompt.strip()), '', gen_text).strip()

    return clean_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html') 


@app.route('/report')
def report():
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    
    report_content = generate_report(lat, lng)
    report_sentences = report_content.split('. ')  # Splitting by sentence
    print(report_sentences)
     # Get weather data
    weather_data = get_weather_data(lat, lng)
    # Get Species
    species = get_species_info(lat,lng)
    species = list(set(species))
    print(species)
    # Generate climate summary
    prediction = generate_climate_summary(weather_data)
    print(prediction)
    return render_template('report.html', report=report_sentences, prediction=prediction, lat=lat, lng=lng, species = species)

if __name__ == "__main__":
    app.run(debug=True)
