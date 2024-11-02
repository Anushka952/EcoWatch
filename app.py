import time
from deep_translator import GoogleTranslator
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer 
import re 
import requests
import torch
from diffusers import StableDiffusionPipeline
from geopy.geocoders import Nominatim



model_id = "CompVis/stable-diffusion-v1-4" 
pipe = StableDiffusionPipeline.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

app = Flask(__name__)


# Image generation using diffusion model
def generate_image_for_species(species_name):
    """Generate an image for a specific species using Stable Diffusion."""
    prompt = f"A highly detailed and realistic depiction of an endangered species, {species_name}, in its natural habitat."
    
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=15).images[0]
        filename = f"static/img/{species_name.replace(' ', '_')}.png" 
        image.save(filename)
    return filename 


# get endangered species name for selected location
def get_species_info(lat, lng):
    species_api_url = f"https://api.gbif.org/v1/occurrence/search?decimalLatitude={lat}&decimalLongitude={lng}"
    response = requests.get(species_api_url)

    response.raise_for_status()
    species_data = response.json()
    species_names = [result['species'] for result in species_data['results']]  

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
    return species_names


# Open-Meteo API endpoint for weather data
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

    daily_data = weather_data.get('daily', {})
    temperature_max = daily_data.get('temperature_2m_max', [])[0] if daily_data.get('temperature_2m_max') else None
    temperature_min = daily_data.get('temperature_2m_min', [])[0] if daily_data.get('temperature_2m_min') else None
    precipitation = daily_data.get('precipitation_sum', [0])[0] if daily_data.get('precipitation_sum') else 0

    summary = (
        f"The maximum temperature is expected to be {temperature_max}°C, "
        f"the minimum temperature is {temperature_min}°C, "
        f"and total precipitation is {precipitation} mm for the selected location."
    )

    # climate summary generation
    model_name = "EleutherAI/gpt-neo-125M"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_text = summary + " The expected overall climate change is,"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(input_ids, max_length=75)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text


# get location name of selected coordinates
def fetch_location(lat, lng):
    geolocator = Nominatim(user_agent="myapp/1.0")
    location_name = ""
    try:
        # Rate limiting
        time.sleep(1)  
        location = geolocator.reverse((lat, lng), exactly_one=True, timeout=10)

        if location:
            location_name = location.raw.get('address', {}).get('suburb') or \
                            location.raw.get('address', {}).get('neighborhood') or \
                            location.raw.get('address', {}).get('city') or \
                            location.raw.get('address', {}).get('county') or \
                            location.raw.get('address', {}).get('state') or \
                            location.raw.get('address', {}).get('country')

            # Translate the location name to English
            if location_name:
                translated_location = GoogleTranslator(source='auto', target='en').translate(location_name)
                return translated_location
        else:
            print("Location not found")
    
    except Exception as e:
        print(f"Error occurred: {e}")
    
    return location_name


# get the coordinates from frontend
@app.route('/set-location', methods=['POST'])
def set_location():
    data = request.get_json()
    lat = data.get('lat')
    lng = data.get('lng')
    
    return jsonify({"status": "success", "latitude": lat, "longitude": lng})


# Action plan generation
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


@app.route('/detail')
def detail():
    return render_template('detail.html') 



@app.route('/report')
def report():
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    print(lat)
    print(lng)
    loc= fetch_location(lat,lng)

    report_content = generate_report(lat, lng)
    report_sentences = report_content.split('. ')
    print(report_sentences)
     # Get weather data
    weather_data = get_weather_data(lat, lng)
    # Get Species data
    species = get_species_info(lat,lng)
    species = list(set(species))
    print(species)
    # Generate climate summary
    prediction = generate_climate_summary(weather_data)
    print(prediction)
    
    image_filenames = [
    "static/img/Dodo_(Raphus_cucullatus).png",
    "static/img/Golden_Toad_(Incilius_periglenes).png",
    "static/img/Heath_Hen_(Tympanuchus_cupido_cupido).png",
    "static/img/Javan_Tiger_(Panthera_tigris_sondaica).png",
    "static/img/Passenger_Pigeon_(Ectopistes_migratorius).png",
    "static/img/Quagga_(Equus_quagga_quagga).png"
    ]

    species = [
         "Raphus_cucullatus",
    "Incilius_periglenes",
    "Tympanuchus_cupido_cupido",
    "Panthera_tigris_sondaica",
    "Ectopistes_migratorius",
    "Equus_quagga"
    ]
    
    # Fetch endangered species data
    # species_list = fetch_species_data(latitude, longitude)
    
    if species:
        print(f"Generating images for {len(species)} endangered species.")
        count = 0
        for species_name in species:
            if count == 0:  
                break
            image_filename = generate_image_for_species(species_name)
            image_filenames.append(image_filename)  # Append the filename to the list
            count += 1
            
        print("Image generation for all species completed.")
    else:
        print("No endangered species found for the given location.")
        
    species_images = [{'filename': img, 'name': name} for img, name in zip(image_filenames, species)]
    return render_template(
        'report.html',
        report=report_sentences,
        prediction=prediction,
        lat=lat,
        lng=lng,
        loc=loc,
        species_images=species_images
    )
    # return render_template('report.html', report=report_sentences, prediction=prediction, lat=lat, lng=lng, species = species,image_filenames=image_filenames)

if __name__ == "__main__":
    app.run(debug=True)
