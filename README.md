# Amharic Braille Reader

An Amharic Braille Reading AI system that converts Amharic Braille characters in images to text.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About

The Amharic Braille Reader is an AI system designed to recognize and convert Amharic Braille characters in images to text. It employs machine learning techniques to process Braille characters and extract meaningful information.

## Features

- **Image Capture and Preprocessing:** Capture and preprocess images containing Amharic Braille characters.
- **Braille Character Segmentation:** Segment individual Braille characters from images for recognition.
- **Character Recognition:** Utilize machine learning models to recognize Braille characters.
- **Word Formation:** Reconstruct words from recognized Braille characters.

## Getting Started

Follow the instructions below to set up the project on your local machine.

### Prerequisites

Make sure you have the following installed:

- Python 3.6 or higher

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/huseny/amharic-braille-reader.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd amharic-braille-reader
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the Amharic Braille Reader, follow these steps:

1. **Capture an Image:** Capture an image containing Amharic Braille characters.

2. **Run the Recognition System:**
    ```bash
    python main.py --input_image path/to/braille_image.jpg
    ```

3. **Review the Output:**
    The system will process the image and output the recognized Amharic text.

## Contributing

Help us improve the Amharic Braille Reader by contributing to its development. Follow the steps below to get started:

1. **Fork the project.**
2. **Create a new branch:**
    ```bash
    git checkout -b feature-branch
    ```
3. **Make your changes and commit them:**
    ```bash
    git commit -m 'Add new feature'
    ```
4. **Push to the branch:**
    ```bash
    git push origin feature-branch
    ```
5. **Submit a pull request.**

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to express our gratitude to the following individuals and projects for their contributions:

- [Contributor Name] for [specific contribution]
- [Recognition Library] for [functionality it provides]
- [Training Dataset] for [relevant data used in training]
