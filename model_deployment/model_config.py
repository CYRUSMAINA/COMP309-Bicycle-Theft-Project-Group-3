"""Shared configuration for model training and Flask inference."""

TARGET_COLUMN = "STATUS"
POSITIVE_STATUS = "RECOVERED"
RANDOM_STATE = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.50

# These fields are available when a theft is reported. PRIMARY_OFFENCE is
# deliberately excluded because values such as "PROPERTY - FOUND" reveal the
# outcome and would leak the target into the model.
NUMERICAL_FEATURES = [
    "OCC_YEAR",
    "OCC_DAY",
    "OCC_HOUR",
    "BIKE_SPEED",
    "BIKE_COST",
    "LONG_WGS84",
    "LAT_WGS84",
]

CATEGORICAL_FEATURES = [
    "OCC_MONTH",
    "OCC_DOW",
    "DIVISION",
    "LOCATION_TYPE",
    "PREMISES_TYPE",
    "BIKE_MAKE",
    "BIKE_TYPE",
    "BIKE_COLOUR",
    "NEIGHBOURHOOD_158",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

FEATURE_DESCRIPTIONS = {
    "OCC_YEAR": "Year when the bicycle theft occurred",
    "OCC_DAY": "Day of the month when the theft occurred (1-31)",
    "OCC_HOUR": "Hour when the theft occurred (0-23)",
    "BIKE_SPEED": "Number of bicycle gears/speeds",
    "BIKE_COST": "Estimated bicycle cost in Canadian dollars",
    "LONG_WGS84": "Approximate occurrence longitude",
    "LAT_WGS84": "Approximate occurrence latitude",
    "OCC_MONTH": "Month when the theft occurred",
    "OCC_DOW": "Day of week when the theft occurred",
    "DIVISION": "Toronto Police Service division",
    "LOCATION_TYPE": "Detailed type of location",
    "PREMISES_TYPE": "Broad premises category",
    "BIKE_MAKE": "Bicycle manufacturer",
    "BIKE_TYPE": "Bicycle type code from the police dataset",
    "BIKE_COLOUR": "Bicycle colour code from the police dataset",
    "NEIGHBOURHOOD_158": "Toronto neighbourhood (158-neighbourhood model)",
}
