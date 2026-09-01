import { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    OCC_YEAR: 2025,
    OCC_DAY: 15,
    OCC_DOY: 200,
    OCC_HOUR: 14,

    REPORT_YEAR: 2025,
    REPORT_DAY: 15,
    REPORT_DOY: 200,
    REPORT_HOUR: 15,

    BIKE_SPEED: 21,
    BIKE_COST: 800,

    LONG_WGS84: -79.38,
    LAT_WGS84: 43.65,

    OCC_MONTH: "July",
    OCC_DOW: "Tuesday",

    REPORT_MONTH: "July",
    REPORT_DOW: "Tuesday",

    DIVISION: "D52",
    LOCATION_TYPE: "Apartment",
    PREMISES_TYPE: "Apartment",

    BIKE_MAKE: "Trek",
    BIKE_TYPE: "Mountain",
    BIKE_COLOUR: "Black",

    PRIMARY_OFFENCE: "THEFT UNDER",

    NEIGHBOURHOOD_158: "Kensington-Chinatown",
    NEIGHBOURHOOD_140: "Kensington-Chinatown",
  });
  const[prediction, setPrediction] = useState("");
  const[error , setError] =useState("");
  const[loading, setLoading]= useState(false);


  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setLoading(true);
    setPrediction("");
    setError("");

    try{
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/predict`,{
          method : "Post",
          headers: {"Content-Type": "application/json",

          },
          body: JSON.stringify({
          ...formData,

          OCC_YEAR: Number(formData.OCC_YEAR),
          OCC_DAY: Number(formData.OCC_DAY),
          OCC_DOY: Number(formData.OCC_DOY),
          OCC_HOUR: Number(formData.OCC_HOUR),

          REPORT_YEAR: Number(formData.REPORT_YEAR),
          REPORT_DAY: Number(formData.REPORT_DAY),
          REPORT_DOY: Number(formData.REPORT_DOY),
          REPORT_HOUR: Number(formData.REPORT_HOUR),

          BIKE_SPEED: Number(formData.BIKE_SPEED),
          BIKE_COST: Number(formData.BIKE_COST),

          LONG_WGS84: Number(formData.LONG_WGS84),
          LAT_WGS84: Number(formData.LAT_WGS84),
        })
        
      }

    );
    const data = await response.json();

    if(!response.ok){
      throw new Error(data.error || "Prediction failed");
    }
    setPrediction(data.prediction);
    
  } catch(error) {
    setError(error.message);
  } finally { 
    setLoading(false);

  }
   
};

  return (
    <div className="app">

      <div className="container">

        <h1>🚲 Bicycle Theft Prediction</h1>

        <p className="subtitle">
          Enter bicycle and incident information to predict the theft status.
        </p>

        <form onSubmit={handleSubmit}>

          {/* ================= INCIDENT INFORMATION ================= */}

          <h2>Incident Information</h2>

          <div className="form-grid">

            <div>
              <label>Occurrence Year</label>
              <input
                type="number"
                name="OCC_YEAR"
                value={formData.OCC_YEAR}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Occurrence Day</label>
              <input
                type="number"
                name="OCC_DAY"
                value={formData.OCC_DAY}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Occurrence Day of Year</label>
              <input
                type="number"
                name="OCC_DOY"
                value={formData.OCC_DOY}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Occurrence Hour</label>
              <input
                type="number"
                name="OCC_HOUR"
                value={formData.OCC_HOUR}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Occurrence Month</label>
              <input
                type="text"
                name="OCC_MONTH"
                value={formData.OCC_MONTH}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Occurrence Day of Week</label>
              <input
                type="text"
                name="OCC_DOW"
                value={formData.OCC_DOW}
                onChange={handleChange}
              />
            </div>

          </div>


          {/* ================= REPORT INFORMATION ================= */}

          <h2>Report Information</h2>

          <div className="form-grid">

            <div>
              <label>Report Year</label>
              <input
                type="number"
                name="REPORT_YEAR"
                value={formData.REPORT_YEAR}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Report Day</label>
              <input
                type="number"
                name="REPORT_DAY"
                value={formData.REPORT_DAY}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Report Day of Year</label>
              <input
                type="number"
                name="REPORT_DOY"
                value={formData.REPORT_DOY}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Report Hour</label>
              <input
                type="number"
                name="REPORT_HOUR"
                value={formData.REPORT_HOUR}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Report Month</label>
              <input
                type="text"
                name="REPORT_MONTH"
                value={formData.REPORT_MONTH}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Report Day of Week</label>
              <input
                type="text"
                name="REPORT_DOW"
                value={formData.REPORT_DOW}
                onChange={handleChange}
              />
            </div>

          </div>


          {/* ================= BICYCLE INFORMATION ================= */}

          <h2>Bicycle Information</h2>

          <div className="form-grid">

            <div>
              <label>Bike Speed</label>
              <input
                type="number"
                name="BIKE_SPEED"
                value={formData.BIKE_SPEED}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Bike Cost</label>
              <input
                type="number"
                name="BIKE_COST"
                value={formData.BIKE_COST}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Bike Make</label>
              <input
                type="text"
                name="BIKE_MAKE"
                value={formData.BIKE_MAKE}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Bike Type</label>
              <input
                type="text"
                name="BIKE_TYPE"
                value={formData.BIKE_TYPE}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Bike Colour</label>
              <input
                type="text"
                name="BIKE_COLOUR"
                value={formData.BIKE_COLOUR}
                onChange={handleChange}
              />
            </div>

          </div>


          {/* ================= LOCATION INFORMATION ================= */}

          <h2>Location Information</h2>

          <div className="form-grid">

            <div>
              <label>Longitude</label>
              <input
                type="number"
                step="any"
                name="LONG_WGS84"
                value={formData.LONG_WGS84}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Latitude</label>
              <input
                type="number"
                step="any"
                name="LAT_WGS84"
                value={formData.LAT_WGS84}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Division</label>
              <input
                type="text"
                name="DIVISION"
                value={formData.DIVISION}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Location Type</label>
              <input
                type="text"
                name="LOCATION_TYPE"
                value={formData.LOCATION_TYPE}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Premises Type</label>
              <input
                type="text"
                name="PREMISES_TYPE"
                value={formData.PREMISES_TYPE}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Neighbourhood 158</label>
              <input
                type="text"
                name="NEIGHBOURHOOD_158"
                value={formData.NEIGHBOURHOOD_158}
                onChange={handleChange}
              />
            </div>

            <div>
              <label>Neighbourhood 140</label>
              <input
                type="text"
                name="NEIGHBOURHOOD_140"
                value={formData.NEIGHBOURHOOD_140}
                onChange={handleChange}
              />
            </div>

          </div>


          {/* ================= OFFENCE ================= */}

          <h2>Offence Information</h2>

          <div className="form-grid">

            <div>
              <label>Primary Offence</label>
              <input
                type="text"
                name="PRIMARY_OFFENCE"
                value={formData.PRIMARY_OFFENCE}
                onChange={handleChange}
              />
            </div>

          </div>


          <button type="submit" disabled = {loading}>
           {loading ? "Predicting...."  : "Predict Bicycle Theft Status"} 
          </button>

        </form>

        {prediction && (
          <div className="result">
            <h2>Prediction</h2>
            <div className="prediction">
              {prediction}
              </div>
              </div>
        )}
        {error &&(
          <div className="error">
            <strong>Error:</strong> {error}
            </div>

        )}

      </div>

    </div>
  );
}

export default App;

