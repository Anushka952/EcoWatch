from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer 

app = Flask(__name__)
CORS(app)


# Load the GPT-2 model and tokenizer globally
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

@app.route('/report', methods=['GET'])
def report():
    lat = request.args.get('lat')
    lng = request.args.get('lng')

    # Create a prompt based on the latitude and longitude
    prompt = f"""
    Create a focused action plan for conserving the habitat at coordinates ({lat}, {lng}). Include a Habitat Assessment, Threat Analysis, measurable Conservation Goals, Actionable Strategies, Community Engagement methods, and a Budget and Resource Plan to ensure sustainable habitat conservation.
    """
    
    # Generate text using the GPT-2 model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=750,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    return jsonify({'report': gen_text})

@app.route('/report')
def report_page():
    return render_template('report.html')

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/report')
# def report():
#     return render_template('report.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/set-location', methods=['POST'])
def set_location():
    data = request.get_json()
    lat = data.get('lat')
    lng = data.get('lng')
    
    print(f"Received coordinates: Latitude={lat}, Longitude={lng}")  # Check if these are not null
    
    return jsonify({"status": "success", "latitude": lat, "longitude": lng})

if __name__ == '__main__':
    app.run(debug=True)
