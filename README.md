# Image Line Thickness Analyzer

This repository contains a Python script for analyzing images to measure the distance between two lines present in the image. It requires that the lines are continupis and purely white. It utilizes image processing techniques to detect the lines and calculates the minimum distance between the lines in both directions. The script also provides visualizations of the image, detected lines, and thickness measurements. The results, including average thickness, standard deviation, minimum and maximum thickness, are printed and saved in a text file. Additionally, an output image with visualizations is generated and saved.

## Installation

To run the Python code, follow these steps:

1. Download the code from this repository or clone it using the following command:
   ```
   git clone https://github.com/li4mm4rtin/smoothMuscleThickness.git
   ```
2. Make sure you have Python 3 installed on your system.
3. Open a terminal or command prompt and navigate to the directory where you downloaded or cloned the code.
4. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

## Usage

To analyze an image and measure line thickness, follow these steps:

1. Open a terminal or command prompt and navigate to the directory where you downloaded or cloned the code.
2. Run the Python script with the following command:
   ```
   python smoothMuscleThickness.py
   ```
3. The script will prompt you to select an image file for analysis. Choose the desired image file.
4. If you want to provide a scale length for accurate measurements, enter the length of the scale bar in the image when prompted. Alternatively, you can manually select the scale bar in the image.
5. The script will process the image, detect the lines, calculate the thickness measurements, and display visualizations.
6. The results, including average thickness, standard deviation, minimum and maximum thickness, will be printed in the terminal or command prompt. The results will also be saved in a text file with the same name as the input image, located in a subdirectory named after the image file.
7. An output image with visualizations will be saved in the same subdirectory as the text file.

Please note that the script requires the OpenCV, NumPy, Matplotlib, Scikit-image, and Tkinter libraries. These dependencies are installed automatically when you run the `pip install -r requirements.txt` command as mentioned in the installation instructions.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository.

## License

This code is licensed under the [MIT License](LICENSE). Feel free to modify and distribute it as needed.
