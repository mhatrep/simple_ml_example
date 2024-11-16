from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.joblib")

# Initialize Flask app
app = Flask(__name__)

# HTML Template for Home Page
HTML_TEMPLATE = """
<!doctype html>
<html>
    <head>
        <title>House Price Prediction</title>
    </head>
    <body>
        <h1>House Price Prediction</h1>
        <form action="/predict" method="post">
            <label for="sqft">Enter Square Footage (comma-separated for multiple):</label>
            <br><br>
            <input type="text" id="sqft" name="sqft" placeholder="e.g., 1500, 2000, 2500">
            <br><br>
            <input type="submit" value="Predict">
        </form>
    </body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    """
    Home page with a simple form for input.
    """
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint that accepts input via:
    1. Form submission (UI)
    2. JSON API (curl or Postman)
    """
    try:
        # If request is from a form submission
        if request.form:
            sqft = request.form.get("sqft")
            if not sqft:
                return "Please provide square footage values.", 400

            # Parse comma-separated values
            sqft = [float(x) for x in sqft.split(",")]
        
        # If request is JSON (from API)
        elif request.is_json:
            data = request.get_json()
            sqft = data.get("sqft")
            if sqft is None:
                return jsonify({"error": "Missing 'sqft' in request"}), 400
            if not isinstance(sqft, list):
                sqft = [sqft]
        
        else:
            return "Unsupported request type. Use form submission or JSON.", 400

        # Convert input to numpy array and make predictions
        sqft = np.array(sqft).reshape(-1, 1)
        predictions = model.predict(sqft).tolist()

        # Return JSON response
        if request.is_json:
            return jsonify({"predictions": predictions})
        
        # Return predictions for form submission
        return f"Predicted Prices: {predictions}"
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
