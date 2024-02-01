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
- [References](#references)

## About

Welcome to the Braille Recognition System repository! This system is designed for the detection and interpretation of Braille characters, specifically tailored for Amharic Braille. The project utilizes a Convolutional Neural Network (CNN) based on the RetinaNet architecture for Braille dot detection. The overall goal is to enhance accessibility for the visually impaired by converting Braille images into text.

## Features

- **Amharic Braille Detection**: The system focuses on detecting Amharic Braille characters with an emphasis on accuracy and robustness.

## Getting Started

Follow the instructions below to set up the project on your local machine.

### Prerequisites

Make sure you have the following installed:

- Python 3.6 or higher

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Huseny/Amharic-Braille-Reader-Backend 
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Amharic-Braille-Reader-Backend
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


## Usage

To use the Amharic Braille Reader, follow these steps:

1. **Run the Backend:**
    ```bash
    python manage.py runserver 0.0.0.0:8000
    ```
2. **Clone the Frontend:**
    ```bash
    git clone https://github.com/Huseny/Amharic-Braille-Reader-Mobile
    ```
3. **Navigate to the project directory:**
    ```bash
    cd Amharic-Braille-Reader-Mobile
    ```
4. **Install dependencies:**
    ```bash
    flutter pub get
    ```
6. **Change the backend address to your backend:**
    go to braille_repository.dart and change the address to your backend address
    ```bash
    static const String _url = 'http://your_backend_address:8000';
    ```
5. **Run the Frontend:**
    ```bash
    flutter run
    ```
6. **Select an Image:**
    Select an image from your gallery or take a picture of a Braille text.

7. **Translate:**
    The system will process the image and output the recognized Amharic text.


## Contributing

Contributions to enhance the system or address issues are welcome. Please fork the repository, create a branch, commit your changes, and submit a pull request:

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

We extend our gratitude to the Ethiopian Association for the Blind and the Addis Ababa University J.F. Kennedy Library for their invaluable support in providing Braille datasets and information. Special thanks to Mr. Abebe for his assistance. We also thank the School of Information Technology and Engineering (SiTE) administration for their support and Professor Xiangdong Wang for insights into the BraUnet.



## References

- [Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
