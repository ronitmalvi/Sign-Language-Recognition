# Sign Language Recognition

This is an innovative application that converts hand gestures into seamless text and speech, bridging the gap between the deaf community and the world.

## Project Structure
```
signrecog/
├── __pycache__/
├── data/
│   └── gesture_model.h5
├── static/
│   └── style.css
├── templates/
│   ├── collect_data.html
│   ├── index.html
│   ├── inference.html
│   └── train_model.html
├── app.py
├── index.py
├── requirements.txt
├── vercel.json
├── wsgi.py
├── X_train.npy
└── y_train.npy
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/transdeaf.git
    cd transdeaf
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Project Files

- [app.py](http://_vscodecontentref_/10): Main Flask application file.
- [index.py](http://_vscodecontentref_/11): Additional application logic.
- `requirements.txt`: List of dependencies.
- `wsgi.py`: WSGI entry point for deployment.
- `templates/`: HTML templates for the web application.
- `static/style.css`: CSS file for styling.
- `data/gesture_model.h5`: Pre-trained model file.
- `X_train.npy`, `y_train.npy`: Training data files.

## Notebooks

- `asl prediction check.ipynb`: Jupyter notebook for ASL prediction checks.
- `set signs to meaning.ipynb`: Jupyter notebook for setting signs to meaning.
- `transdeaf 2.ipynb`: Jupyter notebook for experiments and tests.

## License

This project is licensed under the MIT License. See the [LICENSE](http://_vscodecontentref_/13) file for details.

## Acknowledgements

- MediaPipe for hand gesture recognition.
- TensorFlow and Keras for model training and inference.
