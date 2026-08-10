from ultralytics import YOLO
import os

def export_to_onnx(model_path: str):
    if not os.path.exists(model_path):
        print(f"ERROR: Could not find {model_path}. Please place your trained .pt file here.")
        return

    print(f"Loading PyTorch model from {model_path}...")
    model = YOLO(model_path)

    # Export to ONNX. 
    # dynamic=True allows the model to accept slightly different batch sizes if needed later.
    print("Exporting to ONNX format...")
    onnx_file = model.export(format='onnx', dynamic=True)
    
    print(f"\nSUCCESS! Your highly optimized ONNX model is ready: {onnx_file}")
    print("You can now point InferenceEngine in main.py to this new file.")

if __name__ == "__main__":
    # Replace 'best.pt' with the actual name of your trained YOLO weight file
    export_to_onnx("best.pt")