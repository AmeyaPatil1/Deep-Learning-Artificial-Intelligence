Medical Image Analysis and Classification


Overview


•	Developed deep learning models for medical image classification using PathMNIST (multi-class) and PneumoniaMNIST (binary classification).
•	Built a Streamlit app for interactive image classification.


Datasets

1.	PathMNIST: 
o	Task: Multi-class classification (9 classes).
o	Samples: 89,996 training, 10,004 validation, 7,180 test.
o	Image Size: 28x28x3 (RGB).


3.	PneumoniaMNIST:
   
o	Task: Binary classification (Pneumonia detection).
o	Samples: 4,708 training, 524 validation, 624 test.
o	Image Size: 28x28x1 (Grayscale).
Models


•	PathMNIST:

o	DenseNet121: Pre-trained, high accuracy, rapid convergence.
o	Custom CNN: Tailored architecture for better generalization.


•	PneumoniaMNIST: 

o	ResNet50: Deep residual network for advanced feature learning.
o	Custom CNN: Lightweight and regularized architecture.


Streamlit Application

•	Features: 
o	Upload images for classification.
o	Select models for predictions.
o	Display results with confidence scores.
•	Usage: 
•	streamlit run app.py


Tech Stack

•	Languages: Python
•	Frameworks: TensorFlow, Keras, Streamlit
•	Tools: Jupyter Notebook, Google Colab


Future Enhancements

•	Add multi-label classification tasks.
•	Optimize for faster inference on high-resolution images.
