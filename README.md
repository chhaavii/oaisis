# oAIsis: Real-Time Object Detection

A real-time object detection application using OpenCV and MobileNetSSD. This project can detect common objects through your webcam and provides visual and audio feedback. The name 'oAIsis' combines 'AI' with 'Isis', the Egyptian goddess of wisdom and knowledge, symbolizing the intelligent vision capabilities of this application.

## Features

- Real-time object detection using pre-trained MobileNetSSD model
- Text-to-speech feedback for detected objects
- Simple and clean interface
- Toggle speech output on/off
- Lightweight and easy to set up

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- pyttsx3 (for text-to-speech)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/oAIsis.git
   cd oAIsis
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Or install them manually:
   ```bash
   pip install opencv-python numpy pyttsx3
   ```

3. Download the pre-trained model files if needed:
   - `MobileNetSSD_deploy.prototxt.txt`
   - `MobileNetSSD_deploy.caffemodel`

## Usage

Run the detection script:
```bash
python working_detection.py
```

### Controls
- Press 'q' to quit the application
- Press 's' to toggle speech output on/off

## Files

- `working_detection.py` - Main detection script
- `MobileNetSSD_deploy.prototxt.txt` - Model architecture definition
- `MobileNetSSD_deploy.caffemodel` - Pre-trained model weights
- `requirements.txt` - Python dependencies

## Notes

- The application uses your default webcam (index 0). If you have multiple cameras, you may need to modify the script.
- For best performance, ensure good lighting conditions when running the detection.
- The model can detect 20 different object categories including person, car, chair, etc.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- OpenCV team for the excellent computer vision library
- The creators of MobileNetSSD for the pre-trained model
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

**Step 6:** If you need any help regarding the arguments you pass, try:

```
python real_time_object_detection.py --help
```

### References and Useful Links

* https://github.com/opencv/opencv
* https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
* https://github.com/jrosebr1/imutils
* https://github.com/Surya-Murali/Real-Time-Object-Detection-With-OpenCV
