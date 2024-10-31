from flask import Flask, render_template, request, jsonify
import json
from transformers import AutoModelForCausalLM, AutoTokenizer 
import re 


app = Flask(__name__)



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
        max_length=750,
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
    
    return render_template('report.html', report=report_content, lat=lat, lng=lng)

if __name__ == "__main__":
    app.run(debug=True)
