# RuyaAI
A Simple and high-performance, fully offline Python app for real-time image captioning using BLIP on local GPUs

# Notes
If you have a **CUDA** Device/GPU you need to install pytorch before installing the rest libraries with 
```pip
pip3 install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
```

# Installing Libraries

as simple as running 
```pip
python.exe -m pip install --upgrade pip
```
and then 

```pip
pip install -r requirements.txt
```

# Running Code

```python
python main.py
```

```python
usage: main.py [-h] [--camera CAMERA] [--model-path MODEL_PATH] [--voice VOICE] [--voice-cuda] [--rate-limit RATE_LIMIT] [--headless]

Offline Real-time Camera Captioning

options:
  -h, --help            show this help message and exit
  --camera CAMERA       Camera index
  --model-path MODEL_PATH
                        HuggingFace model ID or local path for BLIP
  --voice VOICE         Path to a Piper .onnx voice model file. Download voices from https://github.com/rhasspy/piper/releases Example: en_US-lessac-medium.onnx
  --voice-cuda          Run Piper voice inference on CUDA (requires onnxruntime-gpu)
  --rate-limit RATE_LIMIT
                        Seconds between captions in auto mode
  --headless            Run without OpenCV window

```

# Usage

inside the program you can run these on your keyboard:

**Q** - Quit the program<br>
**P** - Pause the detection<br> 
**R** - Resume the detection<br>
**A** - Auto mode ( Generate captions and say it every time and time based on RATE_LIMIT )<br>
**M** - Manual mode ( Ganerate captions and say it only when you press G )<br>
**G** - Generate captions and say it in Manual mode<br>
