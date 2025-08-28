# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## Program and output :
```pyhton
import cv2
import numpy as np
import matplotlib.pyplot as plt
faceImage = cv2.imread('photo.jpg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
```
<img width="464" height="522" alt="image" src="https://github.com/user-attachments/assets/81129f14-2cb2-4094-8f61-bd5cfa824184" />

```python
faceImage.shape
glassPNG = cv2.imread('glass.png',-1)
plt.imshow(glassPNG[:,:,::-1])
plt.title("glassPNG")

glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]
```

<img width="765" height="712" alt="image" src="https://github.com/user-attachments/assets/5ad3ad8d-8f58-4d37-a3cf-a13189e7220f" />

```python
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```
<img width="1134" height="238" alt="image" src="https://github.com/user-attachments/assets/3f3b9176-acab-4d18-9cbd-4d786a34a857" />

```python
import cv2
import matplotlib.pyplot as plt

# Load images
faceImage = cv2.imread("Photo.jpg")
glassPNG = cv2.imread("glass.png", -1)

# Extract BGR channels (ignore alpha for naive placement)
glassBGR = glassPNG[:, :, :3]

# Resize glasses to fit
glassBGR = cv2.resize(glassBGR, (250, 80))  # (width, height)

# Copy face
faceWithGlassesNaive = faceImage.copy()

# Define region coordinates properly (y:y+h, x:x+w)
y, x = 300, 305    # top-left corner
h, w = glassBGR.shape[:2]

# Replace ROI with glasses
faceWithGlassesNaive[y:y+h, x:x+w] = glassBGR

# Show result
plt.imshow(faceWithGlassesNaive[..., ::-1])
plt.title("Naive Overlay (No Mask)")
plt.show()
```
<img width="467" height="547" alt="image" src="https://github.com/user-attachments/assets/11e68b0e-e754-4cb7-9398-3348a53fe2dd" />

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your face image
faceImage = cv2.imread("Photo.jpg")
print("Face shape:", faceImage.shape)

# Load sunglasses PNG with alpha channel
glassPNG = cv2.imread("glass.png", -1)   # must have 4 channels
print("Original Glasses shape:", glassPNG.shape)

# Resize sunglasses (adjust size as needed)
glassPNG = cv2.resize(glassPNG, (282, 110))  # width=250, height=80
print("Resized Glasses shape:", glassPNG.shape)

# Separate BGR and Alpha
glassBGR = glassPNG[:, :, :3]
glassMask1 = glassPNG[:, :, 3]

# Convert alpha channel (1-channel) to 3 channels
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Normalize mask (0 or 1 values)
glassMask = np.uint8(glassMask / 255)

# Copy face image
faceWithGlasses = faceImage.copy()

# Define placement (top-left corner of sunglasses)
x, y = 291, 290  # adjust these values for position
h, w = glassBGR.shape[:2]

# Extract Region of Interest (ROI) from the face
roi = faceWithGlasses[y:y+h, x:x+w]

# Masked regions
maskedEye   = cv2.multiply(roi, (1 - glassMask))   # remove sunglass area from ROI
maskedGlass = cv2.multiply(glassBGR, glassMask)    # keep sunglass area

# Combine both
roiFinal = cv2.add(maskedEye, maskedGlass)

# Place combined result back on face
faceWithGlasses[y:y+h, x:x+w] = roiFinal

# Show results
plt.figure(figsize=[15,10])
plt.subplot(121); plt.imshow(faceImage[:,:,::-1]); plt.title("Original")
plt.subplot(122); plt.imshow(faceWithGlasses[:,:,::-1]); plt.title("With Sunglasses (Masked)")
plt.show()
```
<img width="1137" height="747" alt="image" src="https://github.com/user-attachments/assets/3277bde3-3eb6-455b-8d32-bbac5a0fd501" />


Feel free to fork, contribute, or customize this project for your creative needs!
