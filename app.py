from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/set-location', methods=['POST'])
def set_location():
    data = request.get_json()
    lat = data.get('lat')
    lng = data.get('lng')
    
    print(f"Received coordinates: Latitude={lat}, Longitude={lng}")  # Check if these are not null
    
    return jsonify({"status": "success", "latitude": lat, "longitude": lng})

if __name__ == '__main__':
    app.run(debug=True)
