import argparse
import sys
from house_prices import train as train_module
from house_prices import inference as inference_module
import pandas as pd


def Build_model(args):
    """Build the model using a given dataset."""
    print(f"Build model with dataset: {args.input}")
    training_df = pd.read_csv(args.input)
    try:
        train_module.build_model(data=training_df)
        print("Build completed successfully!")
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)


def predict_model(args):
    """Generate predictions and save them to a file."""
    print(f"Generating predictions from: {args.input}")
    predict_df = pd.read_csv(args.input)

    try:
        predictions=inference_module.make_predictions(inference_df=predict_df)

        # Convert predictions to DataFrame
        pred_df = pd.DataFrame({
            "Prediction": predictions
        })

        # Save to CSV
        pred_df.to_csv(args.output, index=False)
        print(f"Predictions {predictions}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="CLI for House Prices ML Project — Train and Predict models easily."
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    #Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--input", required=True, help="Path to training CSV file")
    train_parser.set_defaults(func=Build_model)

    #predict
    predict_parser = subparsers.add_parser("predict", help="Run model inference")
    predict_parser.add_argument("--input", required=True, help="Path to input CSV file")
    predict_parser.add_argument("--output", required=True, help="predictions")
    predict_parser.set_defaults(func=predict_model)

    # Parse and Execute
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
