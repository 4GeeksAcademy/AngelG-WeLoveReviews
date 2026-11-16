"""Run production sentiment inference for the WeLoveReviews reviews CSV."""

from pathlib import Path

import pandas as pd
from transformers import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_with_sentiment.csv"

# Pin to an immutable Hugging Face commit for reproducible inference.
MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"
TEXT_COLUMN = "review_text"
REQUIRED_COLUMNS = {"review_id", "rating", TEXT_COLUMN}
BATCH_SIZE = 16


def predict_sentiment(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
	"""Score review text and write a row-preserving enriched CSV."""
	reviews = pd.read_csv(input_path)
	missing_columns = REQUIRED_COLUMNS - set(reviews.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
	if reviews[TEXT_COLUMN].isna().any():
		raise ValueError(f"Column {TEXT_COLUMN!r} contains missing values")

	# Load the model exactly once, then reuse it for all reviews.
	sentiment_model = pipeline(
		task="sentiment-analysis",
		model=MODEL_ID,
		revision=MODEL_REVISION,
		truncation=True,
	)
	predictions = sentiment_model(
		reviews[TEXT_COLUMN].astype(str).tolist(),
		batch_size=BATCH_SIZE,
	)

	result = reviews.copy()
	result["predicted_stars"] = [int(prediction["label"].split()[0]) for prediction in predictions]
	result["sentiment_band"] = pd.cut(
		result["predicted_stars"],
		bins=[0, 2, 3, 5],
		labels=["Negativa", "Neutral", "Positiva"],
		include_lowest=True,
	).astype(str)
	result["model_confidence"] = [float(prediction["score"]) for prediction in predictions]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(output_path, index=False)
	print(f"Procesadas {len(result)} reseñas.")
	print(f"Salida escrita en: {output_path}")
	print("Distribución de sentimiento:")
	print(result["sentiment_band"].value_counts().reindex(["Positiva", "Neutral", "Negativa"], fill_value=0))
	return result


if __name__ == "__main__":
	predict_sentiment()
