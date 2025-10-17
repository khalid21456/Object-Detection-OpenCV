# Real-Time Object Detection with OpenCV

A Python application that performs real-time object detection using a webcam. Built with OpenCV and SSD MobileNet trained on the COCO dataset, it can detect 80+ different object types including people, vehicles, animals, and everyday items.

## 🎯 Features

- **Real-time detection** through webcam
- **80+ object classes** (COCO dataset)
- **Adjustable confidence threshold** (use +/- keys)
- **Color-coded bounding boxes** for different objects
- **Live statistics** (frame count, detected objects)
- **Screenshot capture** functionality
- **Pause/Resume** capability

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python numpy
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-detection.git
cd object-detection
```

2. Run the application:
```bash
python main.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `S` | Save screenshot |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |
| `P` | Pause/Resume detection |

## 📦 Detectable Objects

The model can detect 80 different object classes including:

**People & Vehicles:**
- person, bicycle, car, motorcycle, bus, truck, train

**Animals:**
- cat, dog, bird, horse, cow, elephant, bear, zebra

**Electronics:**
- laptop, tv, cell phone, keyboard, mouse, remote

**Food & Kitchen:**
- bottle, cup, fork, knife, spoon, banana, apple, pizza

**Furniture & Indoor:**
- chair, couch, bed, dining table, toilet, book, clock

...and many more!

## 📁 Project Structure

```
object-detection/
├── main.py                                    # Main application
├── frozen_inference_graph.pb                  # Pre-trained model
├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt  # Model configuration
├── coco.names                                 # Class labels
└── README.md                                  # This file
```

## 🛠️ Configuration

Adjust these parameters in `main.py`:

```python
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence (0.0 - 1.0)
```

## 📸 Screenshots

*Add your screenshots here showing detection examples*

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

MIT License

## 👤 Author

Your Name - [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- OpenCV for computer vision tools
- TensorFlow for the pre-trained SSD MobileNet model
- COCO dataset for object labels
