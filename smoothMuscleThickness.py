# Author: Liam Martin
# Note: Based on a code originally written by Megan Routzong (Parafin-ManualScale) for calculating smooth muscle
# thickness
# Last Edit: 6/29/23
# Purpose: This code analyzes an image to measure the thickness of lines present in the image. It detects the lines
# using image processing techniques and calculates the minimum distance between the lines in both directions. The code
# also provides visualizations of the image, detected lines, and thickness measurements. The user can provide a scale
# length for accurate measurements, or manually select the scale bar in the image. The results, including average
# thickness, standard deviation, minimum and maximum thickness, are printed and saved in a text file. Additionally,
# an output image with visualizations is generated and saved.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, find_contours
from scipy.spatial.distance import cdist
import tkinter as tk
from tkinter import filedialog
import os


# Gets the path of the image that we are going to analyze (No User Editing Required)
def get_image_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Image File")
    root.destroy()
    return file_path


# Determines the scale length of the image. Takes the user input to learn the length and then has them calculate the
# average of the scale bar's length in pixels (No User Editing Required)
def get_scale_length(image_path, scale_length):
    print("------------------- Getting Pixel Length Data -------------------")
    # Loads the given image
    image = cv2.imread(image_path)

    # Create a window and display the image
    cv2.namedWindow('Image')
    cv2.imshow('Image', image)
    cv2.waitKey(1)

    # Prompt the user to input the scale bar length.
    if scale_length is None:
        scale_length = float(input("Enter the length of the scale bar in the image (in \u03BCm): "))

    # Initialize a list to store the selected points and the list of scale bar pixel lengths
    points = []
    diff_x_list = []

    # Iterate three times to select points for the scale bar ends. Calculate average length
    for i in range(3):
        # Reset points
        points = []

        # Create a copy of the image for marking the points
        marked_image = image.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Draw a crosshair at the clicked point
                cv2.drawMarker(marked_image, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 25)

                # Add the clicked point to the list
                points.append((x, y))

                # Show the marked image
                cv2.imshow('Image', marked_image)
            elif event == cv2.EVENT_RBUTTONDOWN:
                if points:
                    # Remove the last added point
                    points.pop()

                    # Redraw the image without the removed point
                    image_copy = image.copy()

                    for point in points:
                        cv2.drawMarker(image_copy, point, (0, 0, 255), cv2.MARKER_CROSS, 10)

                    cv2.imshow('Image', image_copy)

        # Set the mouse function for point selection
        cv2.setMouseCallback('Image', mouse_callback)

        # Wait for the user to select two points so we can get the length
        while len(points) < 2:
            cv2.imshow('Image', marked_image)
            cv2.waitKey(1)

        # Calculate the difference in x-coordinates
        diff_x = abs(points[0][0] - points[1][0])

        diff_x_list.append(diff_x)

        # Print the difference in x-coordinates
        print(f"Difference in x-coordinates for scale bar end {i + 1}: {diff_x}")

    # Calculate the average length per pixel in the x-direction
    avg_length_per_pixel = scale_length / np.mean([diff_x_list])

    # Close the image window
    cv2.destroyAllWindows()

    print("\n------------------- Pixel Length Data Found and Calculated -------------------")

    return avg_length_per_pixel


# Printer and text save function (No User Editing Required)
def printer(NL, MinDistancesAB, MinDistancesBA, averageLengthPerPixel, MinDistancesLengthAB, MinDistancesLengthBA, FP):
    # Prints some stuff
    print("\n------------------- I am troubleshooting data -------------------\n")
    print(f"Number of detected lines: {NL}")
    print(f"Average thickness 1 -> 2 (pixels):  {np.mean(MinDistancesAB)}")
    print(f"Average thickness 2 -> 1 (pixels):  {np.mean(MinDistancesBA)}")
    print(f"Length in \u03BCm per pixel:  {np.mean(averageLengthPerPixel)}")
    print("\n------------------- I am relevant data -------------------\n")
    print(f"Average thickness 1 -> 2 (\u03BCm): {np.mean(MinDistancesLengthAB)}")
    print(f"Average thickness 2 -> 1 (\u03BCm): {np.mean(MinDistancesLengthBA)} \n")
    print(f"Standard deviation thickness 1 -> 2 (\u03BCm): {np.std(MinDistancesLengthAB)}")
    print(f"Standard deviation thickness 2 -> 1 (\u03BCm): {np.std(MinDistancesLengthBA)} \n")
    print(f"Minimum thickness 1 -> 2 (\u03BCm): {np.min(MinDistancesLengthAB)}")
    print(f"Minimum thickness 2 -> 1 (\u03BCm): {np.min(MinDistancesLengthBA)} \n")
    print(f"Maximum thickness 1 -> 2 (\u03BCm): {np.max(MinDistancesLengthAB)}")
    print(f"Maximum thickness 2 -> 1 (\u03BCm): {np.max(MinDistancesLengthBA)}")

    directory = os.path.dirname(FP)
    file_name = os.path.basename(FP)

    # Create a directory with the same name as the file
    directory_path = os.path.join(directory, os.path.splitext(file_name)[0])
    os.makedirs(directory_path, exist_ok=True)

    # Saves some stuff to a text file.
    textFileName = os.path.join(directory_path, os.path.splitext(file_name)[0] + '.txt')
    try:
        with open(textFileName, 'w') as file:
            file.write("\n------------------- I am troubleshooting data -------------------\n")
            file.write(f"Number of detected lines: {NL}\n")
            file.write(f"Average thickness 1 -> 2 (pixels):  {np.mean(MinDistancesAB)}\n")
            file.write(f"Average thickness 2 -> 1 (pixels):  {np.mean(MinDistancesBA)}\n")
            file.write(f"Length in \u03BCm per pixel:  {np.mean(averageLengthPerPixel)}\n")
            file.write("\n------------------- I am relevant data -------------------\n")
            file.write(f"Average thickness 1 -> 2 (\u03BCm): {np.mean(MinDistancesLengthAB)}\n")
            file.write(f"Average thickness 2 -> 1 (\u03BCm): {np.mean(MinDistancesLengthBA)} \n\n")
            file.write(f"Standard deviation thickness 1 -> 2 (\u03BCm): {np.std(MinDistancesLengthAB)}\n")
            file.write(f"Standard deviation thickness 2 -> 1 (\u03BCm): {np.std(MinDistancesLengthBA)} \n\n")
            file.write(f"Minimum thickness 1 -> 2 (\u03BCm): {np.min(MinDistancesLengthAB)}\n")
            file.write(f"Minimum thickness 2 -> 1 (\u03BCm): {np.min(MinDistancesLengthBA)} \n\n")
            file.write(f"Maximum thickness 1 -> 2 (\u03BCm): {np.max(MinDistancesLengthAB)}\n")
            file.write(f"Maximum thickness 2 -> 1 (\u03BCm): {np.max(MinDistancesLengthBA)}\n")
    except IOError as e:
        print("An error occurred while creating/writing to the file:", str(e))
    print("file written")


# Calculates the minimum distance between the two lines (No User Editing Required)
def calculate_min_distances(line1, line2):
    print("------------------- Finding Friends -------------------")
    distances = cdist(line1, line2)
    min_distances = np.min(distances, axis=1)
    min_indices = np.argmin(distances, axis=1)
    print("------------------- Friends Found -------------------")
    return min_distances, min_indices


# Creates and displays the images of the two lines. Also calls the functions to find the thickness
def create_lines_image(image_path, lineThickness, ALPP):
    # Detects the lines on the image
    print("------------------- Detecting Lines -------------------")
    lineThickness = 5
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_preExapnasion = cv2.threshold(grayscale, 254, 255, cv2.THRESH_BINARY)
    threshold_preExapnasion = cv2.bitwise_not(threshold_preExapnasion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (lineThickness, lineThickness))
    threshold = cv2.erode(threshold_preExapnasion, kernel)
    labeled_image = label(threshold, connectivity=2)
    contours = find_contours(labeled_image, 0.5, fully_connected='low', positive_orientation='low')
    print("------------------- Lines Detected -------------------\n")

    # Ensures the the correct number of lines have been found. Then finds the distance between the two in both
    # directions
    if len(contours) == 2:
        print("------------------- Finding Distance From A to B -------------------")
        minimumDistancesAB, minIndexAB = calculate_min_distances(contours[0], contours[1])

        print("------------------- Found Distance From A to B -------------------\n")

        print("------------------- Finding Distance From B to A -------------------")
        minimumDistancesBA, minIndexBA = calculate_min_distances(contours[1], contours[0])
        print("------------------- Found Distance From B to A -------------------")
    elif len(contours) > 2:
        print(len(contours))
        exit("Too Many Contours: Increase Line Thickness")
    else:
        exit("Too Few Contours: Decrease Line Thickness")

    # Creates and displays the figure
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 3)
    axs1 = fig.add_subplot(gs[0, 0])
    axs2 = fig.add_subplot(gs[0, 1])
    axs3 = fig.add_subplot(gs[0, 2])
    axs4 = fig.add_subplot(gs[1, 0])
    axs5 = fig.add_subplot(gs[1, 1])
    axs6 = fig.add_subplot(gs[1, 2])
    axs7 = fig.add_subplot(gs[2, :])

    # Shows the input image
    axs1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs1.set_title('Original Image')
    axs1.axis('off')

    # Shows the grayscale version
    axs2.imshow(grayscale, cmap='gray')
    axs2.set_title('Grayscale Image')
    axs2.axis('off')

    # Shows the detected lines
    axs3.imshow(threshold_preExapnasion, cmap='gray')
    axs3.set_title('Threshold Image')
    axs3.axis('off')

    # Shows the expanded lines
    axs4.imshow(threshold, cmap='gray')
    axs4.set_title('Threshold Image (Thickened)')
    axs4.axis('off')

    # Overlays the splines on the image
    axs5.imshow(grayscale, cmap='gray')
    axs5.set_title('Overlay of Detected Lines and Gray Image')
    axs5.axis('off')

    colors = plt.cm.tab10(np.linspace(0, 1, len(contours)))
    for i, (spline, color) in enumerate(zip(contours, colors)):
        axs5.plot(spline[:, 1], spline[:, 0], color=color)
        axs5.text(spline[0, 1], spline[0, 0], f'{i + 1}', color='white', fontsize=14)

    # Shows the lines used to measure thickness
    axs6.imshow(grayscale, cmap='gray')
    axs6.set_title('Thickness Measurement Lines')
    axs6.axis('off')

    for i in range(len(contours[0])):
        axs6.plot([contours[0][i, 1], contours[1][minIndexAB[i], 1]],
                  [contours[0][i, 0], contours[1][minIndexAB[i], 0]],
                  color='red', alpha=0.025)

    for i in range(len(contours[1])):
        axs6.plot([contours[1][i, 1], contours[0][minIndexBA[i], 1]],
                  [contours[1][i, 0], contours[0][minIndexBA[i], 0]],
                  color='orange', alpha=0.025)

    for i, (spline, color) in enumerate(zip(contours, colors)):
        axs6.plot(spline[:, 1], spline[:, 0], color=color)
        axs6.text(spline[0, 1], spline[0, 0], f'{i + 1}', color='white', fontsize=14)

    minimumDistancesLengthAB = minimumDistancesAB * ALPP
    minimumDistancesLengthBA = minimumDistancesBA * ALPP

    # Shows the histogram of the measurement
    axs7.hist(minimumDistancesLengthAB, label='Thickness from 1 -> 2', alpha=0.5, color='red')
    axs7.axvline(x=np.mean(minimumDistancesLengthAB), label='Mean thickness 1 -> 2', linestyle='dashed', color='red')
    axs7.hist(minimumDistancesLengthBA, label='Thickness from 1 -> 2', alpha=0.5, color='orange')
    axs7.axvline(x=np.mean(minimumDistancesLengthBA), label='Mean thickness 2 -> 1', linestyle='dashed', color='orange')
    axs7.set_title('Histograms of Measurement Lines')
    axs7.legend()
    axs7.set_ylabel('Count')
    axs7.set_xlabel('Thickness (\u03BCm)')

    print(len(minimumDistancesLengthAB), len(minimumDistancesLengthBA))

    # Calls printer and saves text file
    printer(
        len(contours),
        minimumDistancesAB,
        minimumDistancesBA,
        ALPP,
        minimumDistancesLengthAB,
        minimumDistancesLengthBA,
        image_path
    )

    directory = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)

    # Create a directory with the same name as the file
    directory_path = os.path.join(directory,os.path.splitext(file_name)[0])
    print(directory_path)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            directory_path,
            os.path.splitext(file_name)[0] + '_output.png'))
    plt.show()


# IF you are running a bunch with a single scale length you can change it here. If you want to manually do this for
# every file please type SL = NONE, otherwise type the length of the scale in the form SL = 1000

SL = None
# SL = 1000

image_path = get_image_path()
averageLengthPerPixel = get_scale_length(image_path, scale_length=SL)
create_lines_image(image_path, lineThickness=5, ALPP=averageLengthPerPixel)
