import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL.PngImagePlugin import PngInfo
import math
import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import cv2
import matplotlib.pyplot as plt


def unpack_lines_data(metadata):
    lines = []
    lines_data = metadata

    if lines_data:
        lines_data = lines_data.split("|")
        for line_data in lines_data:
            points = [tuple(map(float, point.split(","))) for point in line_data.split(";")]
            lines.append(points)

    # for i in range(len(lines)):
    #     print(lines[i])
    #     lines[i] = np.array(lines[i])

    return lines


# https://stackoverflow.com/questions/67460967/cubic-spline-for-non-monotonic-data-not-a-1d-function
def convert_to_spline(linedata, n_index=1000):
    x = np.array(linedata)[:, 0]
    y = np.array(linedata)[:, 1]

    t = np.linspace(0, 1, x.size)
    r = np.vstack((x.reshape((1, x.size)), y.reshape((1, y.size))))

    spline = interp1d(t, r, kind='cubic')
    t = np.linspace(np.min(t), np.max(t), n_index)

    return np.transpose(spline(t))


def calculate_min_distance(lines1, lines2):
    print("------------------- Finding Friends -------------------")
    distances = cdist(lines1, lines2)
    print(distances.shape)
    min_distances = np.min(distances, axis=1)
    min_indices = np.argmin(distances, axis=1)
    print("------------------- Friends Found -------------------")
    print(min_distances.shape, min_indices.shape)
    return min_distances, min_indices


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
            if NL == 4:
                file.write("Assumed complete circle in image. If this is not desired effect please adjust image.")
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


class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")

        self.canvas = tk.Canvas(root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = None
        self.file_path = None
        self.original_point = (None, None)
        self.min_distances = np.empty(1)
        self.min_indices = np.empty(1)
        self.current_line = []
        self.lines = []
        self.splines = self.lines
        self.editing_mode = False
        self.editing_line = None
        self.dragged_point = None
        self.selected_point = None
        self.scale_length = None
        self.avg_length_per_pixel = -1000.0
        self.is_donut = False

        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.root.bind("<Return>", self.on_enter_key)
        self.root.bind("<Escape>", self.on_escape_key)

        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Save Image with Lines", command=self.save_image_with_lines)
        file_menu.add_separator()
        file_menu.add_checkbutton(label="is_donut", variable=self.is_donut, command=self.toggle_is_donut)
        menubar.add_cascade(label="File", menu=file_menu)
        root.config(menu=menubar)

        self.delete_mode = False
        self.delete_button = tk.Button(root, text="Delete Mode", command=self.toggle_delete_mode)
        self.delete_button.pack()

        self.edit_mode = False
        self.edit_button = tk.Button(root, text="Edit Mode", command=self.toggle_edit_mode)
        self.edit_button.pack()

        self.measure_button = tk.Button(root, text="Measure", command=self.create_lines_image)
        self.measure_button.pack()

        self.scale_button = tk.Button(root, text="Determine Scale", command=self.determine_pixel_scale)
        self.scale_button.pack()

    def open_image(self):
        # self.file_path = "./output_images/combined_lines_image.png"
        # self.file_path = "./testImages/testImage_noLines.png"
        self.file_path = filedialog.askopenfilename(initialdir='./')
        if self.file_path:
            self.image = Image.open(self.file_path)
            # Load lines data from the metadata, if available
            # Get the dimensions of the loaded image
            width, height = self.image.size
            # Update the Tkinter window size based on image dimensions
            self.root.geometry(f"{width}x{height + 100}")
            if self.image.text.get("Lines"):
                self.lines = unpack_lines_data(self.image.text["Lines"])
            if self.image.text.get("PixelLength"):
                self.avg_length_per_pixel = float(self.image.text["PixelLength"])
                print(self.avg_length_per_pixel)
            # Update the canvas with the loaded image and lines
            self.update_image()

    def save_image_with_lines(self):
        if self.image is not None:
            # Create a metadata chunk to store the lines data
            lines_chunk = PngInfo()

            # Convert lines data to a string representation and add it to the metadata
            lines_data = "|".join(";".join(",".join(map(str, point)) for point in line) for line in self.lines)
            lines_chunk.add_text("Lines", lines_data)

            lines_chunk.add_text("PixelLength", str(self.avg_length_per_pixel))

            # Ask the user for the file name and location to save the image
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])

            if file_path:
                # Save the edited image with the lines metadata at the user-specified location
                edited_image = self.image
                edited_image.save(file_path, pnginfo=lines_chunk)

    def update_image(self):
        if self.image is not None:
            image_tk = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
            self.canvas.image = image_tk  # Save a reference to prevent garbage collection
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            if not self.lines == [[-1000]]:
                self.draw_lines()
                self.draw_points()

    def on_canvas_left_click(self, event):
        if self.image is not None and not self.edit_mode:
            x, y = event.x, event.y
            if len(self.current_line) == 0:
                self.current_line.append((x, y))
            self.current_line.append((x, y))
            self.draw_line_segment(self.current_line[-2], self.current_line[-1], "red", 2)
        elif self.edit_mode:
            x, y = event.x, event.y
            line, index = self.find_nearest_point(x, y)
            if line is not None:
                self.selected_point = (line, index)
                self.editing_line = line
                self.dragged_point = index
                self.original_point = (x, y)  # Save the original coordinates of the edited point
                self.draw_points()

    def on_canvas_drag(self, event):
        if self.current_line and not self.editing_mode:
            x, y = event.x, event.y
            self.canvas.coords(self.current_line[-1], x, y)
        elif self.editing_line and self.dragged_point is not None:
            x, y = event.x, event.y
            dx, dy = x - self.original_point[0], y - self.original_point[1]
            self.editing_line[self.dragged_point] = (self.original_point[0] + dx, self.original_point[1] + dy)
            self.original_point = (x, y)  # Update the original point for smooth dragging
            self.draw_lines()  # Redraw the entire canvas from scratch
            self.draw_points()

    def on_canvas_right_click(self, event):
        if self.image is not None and self.edit_mode:
            x, y = event.x, event.y
            line, index = self.find_nearest_point(x, y)
            if line is not None:
                self.selected_point = (line, index)
                self.editing_line = line
                self.dragged_point = index
                self.original_point = (x, y)  # Save the original coordinates of the edited point
                self.draw_points()

    def on_canvas_drag(self, event):
        if self.current_line and not self.editing_mode:
            x, y = event.x, event.y
            self.canvas.coords(self.current_line[-1], x, y)
        elif self.editing_line and self.dragged_point is not None:
            x, y = event.x, event.y
            dx, dy = x - self.original_point[0], y - self.original_point[1]
            self.editing_line[self.dragged_point] = (self.original_point[0] + dx, self.original_point[1] + dy)
            self.draw_lines()  # Redraw the entire canvas from scratch
            self.draw_points()

    def draw_line_segment(self, start, end, color, width):
        return self.canvas.create_line(start, end, fill=color, width=width)

    def draw_lines(self):
        self.canvas.delete("all")  # Clear the entire canvas
        if self.image is not None:
            image_tk = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
            self.canvas.image = image_tk  # Save a reference to prevent garbage collection

        for line in self.lines:
            for i in range(1, len(line)):
                self.draw_line_segment(line[i - 1], line[i], "blue", 2)

            if self.is_donut and len(line) >= 2:
                # If is_donut is True, connect the first and last points of the line
                self.draw_line_segment(line[0], line[-1], "blue", 2)

        if self.editing_mode and self.current_line:
            for i in range(1, len(self.current_line)):
                self.draw_line_segment(self.current_line[i - 1], self.current_line[i], "red", 2)

    def draw_points(self):
        self.canvas.delete("point")
        for line in self.lines:
            for x, y in line:
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="blue", tags="point")
        if self.current_line:
            for x, y in self.current_line:
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", tags="point")
        if self.selected_point:
            line, index = self.selected_point
            x, y = line[index]
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="green", tags="point")

    def on_enter_key(self, event):
        if len(self.current_line) > 1:
            self.lines.append(list(self.current_line))
            self.current_line = []
            self.editing_line = None
            self.selected_point = None
            self.draw_lines()
            self.draw_points()

    def on_escape_key(self, event):
        self.current_line = []
        self.editing_line = None
        self.selected_point = None
        self.draw_lines()
        self.draw_points()

    def toggle_is_donut(self):
        self.is_donut = not self.is_donut

    def find_nearest_point(self, x, y):
        nearest_distance = float("inf")
        nearest_line = None
        nearest_index = None

        for line in self.lines:
            for index, (px, py) in enumerate(line):
                distance = math.sqrt((px - x) ** 2 + (py - y) ** 2)
                if distance < nearest_distance and distance < 100:  # Only consider points within 5 pixels
                    nearest_distance = distance
                    nearest_line = line
                    nearest_index = index

        return nearest_line, nearest_index

    def toggle_delete_mode(self):
        if self.edit_mode:
            self.turn_off_edit_mode()
        self.delete_mode = not self.delete_mode
        if self.delete_mode:
            self.turn_on_delete_mode()
        else:
            self.turn_off_edit_mode()

    def turn_off_delete_mode(self):
        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.delete_button.config(text="Delete Mode")

    def turn_on_delete_mode(self):
        self.canvas.bind("<Button-1>", self.delete_line)
        self.delete_button.config(text="Stop Delete")

    def toggle_edit_mode(self):
        if self.delete_mode:
            self.turn_off_delete_mode()
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.turn_on_edit_mode()
        else:
            self.turn_off_edit_mode()

    def turn_off_edit_mode(self):
        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.edit_button.config(text="Edit Mode")
        self.selected_point = None
        self.editing_line = None
        self.dragged_point = None

    def turn_on_edit_mode(self):
        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.edit_button.config(text="Stop Edit")

    def delete_line(self, event):
        if self.delete_mode and not self.edit_mode:
            x, y = event.x, event.y
            line, _ = self.find_nearest_point(x, y)
            if line is not None:
                self.lines.remove(line)
                self.draw_lines()
                self.draw_points()

    def determine_pixel_scale(self):
        print("------------------- Getting Pixel Length Data -------------------")
        # Loads the given image
        image = cv2.imread(self.file_path)

        # Create a window and display the image
        cv2.namedWindow('Image')
        cv2.imshow('Image', image)
        cv2.waitKey(1)

        # Prompt the user to input the scale bar length.
        if self.scale_length is None:
            self.scale_length = tk.simpledialog.askfloat("Input Number", "Please enter a number:")

        cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)

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
        self.avg_length_per_pixel = self.scale_length / np.mean([diff_x_list])

        # Close the image window
        cv2.destroyAllWindows()

        print("\n------------------- Pixel Length Data Found and Calculated -------------------")

    def create_lines_image(self):
        # Detects the lines on the image
        image = cv2.imread(self.file_path)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for i, line in enumerate(self.lines):
            self.lines[i] = np.array(line)

        print(self.lines[0])

        self.splines[0] = convert_to_spline(self.lines[0])
        self.splines[1] = convert_to_spline(self.lines[1])

        print(self.lines[0])

        print("------------------- Finding Distance From A to B -------------------")
        minimumDistancesAB, minIndexAB = calculate_min_distance(self.splines[0], self.splines[1])
        print("------------------- Found Distance From A to B -------------------\n")

        print("------------------- Finding Distance From B to A -------------------")
        minimumDistancesBA, minIndexBA = calculate_min_distance(self.splines[1], self.splines[0])
        print("------------------- Found Distance From B to A -------------------")

        # Creates and displays the figure
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(3, 2)
        axs1 = fig.add_subplot(gs[0, 0])
        axs2 = fig.add_subplot(gs[0, 1])
        axs5 = fig.add_subplot(gs[1, 0])
        axs6 = fig.add_subplot(gs[1, 1])
        axs7 = fig.add_subplot(gs[2, :])

        # Shows the input image
        axs1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs1.set_title('Original Image')
        axs1.axis('off')

        # Shows the grayscale version
        axs2.imshow(grayscale, cmap='gray')
        axs2.set_title('Grayscale Image')
        axs2.axis('off')

        # Overlays the splines on the image
        axs5.imshow(grayscale, cmap='gray')
        axs5.set_title('Overlay of Detected Lines and Gray Image')
        axs5.axis('off')

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.splines)))
        for i, (spline, color) in enumerate(zip(self.splines, colors)):
            axs5.plot(spline[:, 0], spline[:, 1], color=color)
            axs5.text(spline[0, 0], spline[0, 1], f'{i + 1}', color='white', fontsize=14)

        # Shows the lines used to measure thickness
        axs6.imshow(grayscale, cmap='gray')
        axs6.set_title('Thickness Measurement Lines')
        axs6.axis('off')

        for i in range(len(self.splines[0])):
            axs6.plot([self.splines[0][i, 0], self.splines[1][minIndexAB[i], 0]],
                      [self.splines[0][i, 1], self.splines[1][minIndexAB[i], 1]],
                      color='red', alpha=0.25)

        for i in range(len(self.lines[1])):
            axs6.plot([self.splines[1][i, 0], self.splines[0][minIndexBA[i], 0]],
                      [self.splines[1][i, 1], self.splines[0][minIndexBA[i], 1]],
                      color='orange', alpha=0.25)

        for i, (spline, color) in enumerate(zip(self.splines, colors)):
            axs6.plot(spline[:, 0], spline[:, 1], color=color)
            axs6.text(spline[0, 0], spline[0, 1], f'{i + 1}', color='white', fontsize=14)

        minimumDistancesLengthAB = minimumDistancesAB * self.avg_length_per_pixel
        minimumDistancesLengthBA = minimumDistancesBA * self.avg_length_per_pixel

        # Shows the histogram of the measurement
        axs7.hist(minimumDistancesLengthAB, label='Thickness from 1 -> 2', alpha=0.5, color='red')
        axs7.axvline(x=np.mean(minimumDistancesLengthAB), label='Mean thickness 1 -> 2', linestyle='dashed',
                     color='red')
        axs7.hist(minimumDistancesLengthBA, label='Thickness from 2 -> 1', alpha=0.5, color='orange')
        axs7.axvline(x=np.mean(minimumDistancesLengthBA), label='Mean thickness 2 -> 1', linestyle='dashed',
                     color='orange')
        axs7.set_title('Histograms of Measurement Lines')
        axs7.legend()
        axs7.set_ylabel('Count')
        axs7.set_xlabel('Thickness (\u03BCm)')

        # Calls printer and saves text file
        printer(
            len(self.lines),
            minimumDistancesAB,
            minimumDistancesBA,
            self.avg_length_per_pixel,
            minimumDistancesLengthAB,
            minimumDistancesLengthBA,
            self.file_path
        )

        directory = os.path.dirname(self.file_path)
        file_name = os.path.basename(self.file_path)

        # Create a directory with the same name as the file
        directory_path = os.path.join(directory, os.path.splitext(file_name)[0])
        print(directory_path)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                directory_path,
                os.path.splitext(file_name)[0] + '_output.png'))
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
