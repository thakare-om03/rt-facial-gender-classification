import argparse
from src.train import train_model
from src.predict import predict_single, predict_group

def main():
    parser = argparse.ArgumentParser(description="Gender Classification Application")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Subparser for model training
    train_parser = subparsers.add_parser('train', help='Train the gender classification model')

    # Subparser for prediction (single face or group photo)
    predict_parser = subparsers.add_parser('predict', help='Predict gender from an image')
    predict_parser.add_argument('--mode', type=str, choices=['single', 'group'], default='single',
                                help="Prediction mode: 'single' for one face, 'group' for all faces")
    predict_parser.add_argument('image_path', type=str, help='Path to the image file')

    args = parser.parse_args()

    if args.command == 'train':
        train_model()
    elif args.command == 'predict':
        if args.mode == 'single':
            gender = predict_single(args.image_path)
            if gender:
                print("Predicted Gender:", gender)
        else:
            results = predict_group(args.image_path)
            if results:
                for bbox, gender in results:
                    print(f"Face at {bbox} predicted as {gender}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
