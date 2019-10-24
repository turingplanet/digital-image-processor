import os, numpy, cv2
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

import tkinter.messagebox
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from functools import partial

import img_processor

class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.initialization()
    
    def initialization(self):
        # Default values 
        self.img_path = "./static/lena512.pbm"
        self.img_array = numpy.array(Image.open(self.img_path))
        self.functionality_frame_width, self.functionality_frame_height = 450, 50
        self.image_frame_width, self.image_frame_height = 780, 700
        # Set up frames
        self.initialize_menu()
        self.initialize_image_resize_frame(Frame(self))
        self.initialize_gray_level_frame(Frame(self))
        self.initialize_histogram_equalization_frame(Frame(self))
        self.initialize_spatial_filtering_frame(Frame(self))
        self.initialize_bit_panel_frame(Frame(self))
        self.initialize_noise_generating_frame(Frame(self))
        self.initialize_restoration_spatial_filtering_frame(Frame(self))
        self.initialize_image_helper_frame(Frame(self))
        self.initialize_zoom_shrink_frame(Frame(self))
        self.initialize_image_frame(Frame(self, highlightbackground="black", highlightthickness=1))
        self.master.geometry(f"{self.functionality_frame_width + self.image_frame_width}x{self.image_frame_height}")
    
    '''
    Build Menu
    '''
    def initialize_menu(self):
        menu = Menu(self)
        self.master.config(menu=menu)
        fileMenu = Menu(menu)
        menu.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="New Image", command=self.open_image)
        fileMenu.add_command(label="Save Image", command=self.save_image) 

    def open_image(self):
        # choose an new image path
        new_img_path =  filedialog.askopenfilename(initialdir = "./static",title = "Select file",filetypes = (
            ("pbm file", "*.pbm"), 
            ("all files","*.*"),
            ("jpeg files","*.jpg")))
        if new_img_path:
            self.img_path = new_img_path
            # Update the new image display
            self.ori_img = Image.open(self.img_path)
            self.img_width, self.img_height = self.ori_img.size
            self.img_array = numpy.array(Image.open(self.img_path))
            self.new_width, self.new_height = self.ori_img.size
            self.update_image(numpy.array(self.ori_img))
            # Update width and heighter contoller and bits controller
            self.width_input.delete(1.0, END)
            self.width_input.insert(END, self.new_width)
            self.height_input.delete(1.0, END)
            self.height_input.insert(END, self.new_height)

    def save_image(self):
        popup_save_window = Toplevel()
        popup_save_window.geometry('400x50')
        popup_save_window.wm_title("Save Image")
        file_name_label = Label(popup_save_window, text="New File Name: ")
        file_name_label.pack(padx=10, side=LEFT)
        new_file_name_entry = Entry(popup_save_window, width=12, textvariable=StringVar(value='new_file.pbm'), highlightbackground='black', highlightthickness=1)
        new_file_name_entry.pack(padx=10, side=LEFT)
        save_image_button = Button(popup_save_window, height=1, text='Save Image', command=partial(self.popup_save_image, popup_save_window, new_file_name_entry))
        save_image_button.pack(padx=10, side=RIGHT)

    def popup_save_image(self, popup_save_window, new_file_name_entry):
        new_file_name = new_file_name_entry.get()
        popup_save_window.destroy()
        if not os.path.exists("./static/new_images"):
            os.mkdir('./static/new_images')
        (Image.fromarray(self.modified_img_array)).save("./static/new_images/" + new_file_name)
        messagebox.showinfo(title="Image Saved", message=f"You saved new image in \nstatic/new_images/{new_file_name}")
    
    '''
    Build image resizing frame
    '''
    def initialize_image_resize_frame(self, image_resize_frame):
        image_resize_frame.place(x=0, y=0, width=self.functionality_frame_width, height=self.functionality_frame_height)
        # Initialize algorithms drop down menu and resize button
        self.zooming_algorithm = StringVar(image_resize_frame)
        self.zooming_algorithm.set("Nearest Neighbor") # default value
        algorithms = ['Nearest Neighbor', 'Linear Method (x)', 'Linear Method (y)', 'Bilinear Interpolation', 'PIL Library']
        self.zooming_algorithm_input = OptionMenu(image_resize_frame, self.zooming_algorithm, algorithms[0], algorithms[1], algorithms[2], algorithms[3], algorithms[4])
        self.zooming_algorithm_input.pack(side=LEFT)
        # Set default width and height in width and height input
        self.ori_img = Image.open(self.img_path)
        self.new_width, self.new_height = self.ori_img.size
        width_label = Label(image_resize_frame, text="width").pack(side=LEFT)
        self.width_input = Text(image_resize_frame, width=4, height=1, highlightbackground='black', highlightthickness=1)
        self.width_input.pack(side=LEFT)
        self.width_input.insert(END, self.new_width)
        height_label = Label(image_resize_frame, text="height").pack(side=LEFT)
        self.height_input = Text(image_resize_frame, width=4, height=1, highlightbackground='black', highlightthickness=1)
        self.height_input.pack(side=LEFT)
        self.height_input.insert(END, self.new_height)
        resize_button = Button(image_resize_frame, text="Resize", command=self.resize_image)
        resize_button.pack(side=LEFT)

    def resize_image(self):
        algorithm = self.zooming_algorithm.get()
        self.new_width = int(self.width_input.get('1.0', END))
        self.new_height = int(self.height_input.get('1.0', END))
        if algorithm == 'Nearest Neighbor':
            new_img_array = img_processor.nearestNeighbor(self.img_array, self.new_width, self.new_height)
            self.update_image(new_img_array)
        elif algorithm == 'Linear Method (x)':
            new_img_array = img_processor.linearX(self.img_array, self.new_width, self.new_height)
            self.update_image(new_img_array)
        elif algorithm == 'Linear Method (y)':
            new_img_array = img_processor.linearY(self.img_array, self.new_width, self.new_height)
            self.update_image(new_img_array)
        elif algorithm == 'Bilinear Interpolation':
            new_img_array = img_processor.bilinear(self.img_array, self.new_width, self.new_height)
            self.update_image(new_img_array)
        elif algorithm == 'PIL Library':
            new_img_array = numpy.array(Image.fromarray(self.img_array).resize((self.new_width, self.new_height)))
            self.update_image(new_img_array)

    ''' 
    Build gray level modifying frame
    '''
    def initialize_gray_level_frame(self, gray_level_frame):
        gray_level_frame.place(x=0, y=self.functionality_frame_height, width=self.functionality_frame_width, height=self.functionality_frame_height)

        self.gray_level = StringVar(gray_level_frame)
        self.gray_level.set("8") # default value
        self.gray_level_input = OptionMenu(gray_level_frame, self.gray_level, "1", "2", "3", "4", "5", "6", "7", "8")
        self.gray_level_input.pack(side=LEFT)

        gray_level_label = Label(gray_level_frame, text="Bits").pack(side=LEFT)

        gray_level_button = Button(gray_level_frame, text="Change gray level", command=self.change_gray_level)
        gray_level_button.pack(side=LEFT)

    def change_gray_level(self):
        gray_level = self.gray_level.get()
        new_img_array = img_processor.convertGrayLevel(self.img_array, 8, int(gray_level))
        self.update_image(new_img_array)

    '''
    Build histogram equalization frame
    '''
    def initialize_histogram_equalization_frame(self, histogram_equalization_frame):
        histogram_equalization_frame.place(x=0, y=self.functionality_frame_height * 2, width=self.functionality_frame_width, height=self.functionality_frame_height)
        # Initialize histogram equalization drop down menu and resize button
        self.histogram_equalization_choice = StringVar(histogram_equalization_frame)
        self.histogram_equalization_choice.set("Global") # default value
        histogram_equalization_options = ['Global', 'Local']
        self.histogram_equalization_menu = OptionMenu(histogram_equalization_frame, self.histogram_equalization_choice, histogram_equalization_options[0], histogram_equalization_options[1], 
            command=self.histogram_equalization_option_menu_handler)
        self.histogram_equalization_menu.pack(side=LEFT)

        width_label = Label(histogram_equalization_frame, text="width").pack(side=LEFT)
        self.histogram_equalization_mask_width = Text(histogram_equalization_frame, width=3, height=1, 
            highlightbackground='black', highlightthickness=1, background="gray")
        self.histogram_equalization_mask_width.pack(side=LEFT)
        self.histogram_equalization_mask_width.insert(END, 3)

        height_label = Label(histogram_equalization_frame, text="height").pack(side=LEFT)
        self.histogram_equalization_mask_height = Text(histogram_equalization_frame, width=3, height=1, 
            highlightbackground='black', highlightthickness=1, background="gray")
        self.histogram_equalization_mask_height.pack(side=LEFT)
        self.histogram_equalization_mask_height.insert(END, 3)

        # Initially, mask width and height input are disabled
        self.histogram_equalization_mask_width.config(state=DISABLED)
        self.histogram_equalization_mask_height.config(state=DISABLED)

        resize_button = Button(histogram_equalization_frame, text="Histogram Equalization", command=self.histogram_equalization)
        resize_button.pack(side=LEFT)

    def histogram_equalization_option_menu_handler(self, *args):
        if self.histogram_equalization_choice.get() == 'Global':
            self.histogram_equalization_mask_width.config(state=DISABLED)
            self.histogram_equalization_mask_width.config(background='gray')
            self.histogram_equalization_mask_height.config(state=DISABLED)
            self.histogram_equalization_mask_height.config(background='gray')
        elif self.histogram_equalization_choice.get() == 'Local':
            self.histogram_equalization_mask_width.config(state=NORMAL)
            self.histogram_equalization_mask_width.config(background='white')
            self.histogram_equalization_mask_height.config(state=NORMAL)
            self.histogram_equalization_mask_height.config(background='white')
    
    def histogram_equalization(self):
        if self.histogram_equalization_choice.get() == 'Global':
            new_img_array = img_processor.global_histogram_equalization(self.img_array)
            self.update_image(new_img_array)
        elif self.histogram_equalization_choice.get() == 'Local':
            new_img_array = img_processor.local_histogram_equalization(self.img_array, 
                int(self.histogram_equalization_mask_width.get('1.0', END)), int(self.histogram_equalization_mask_height.get('1.0', END)))
            self.update_image(new_img_array)

    '''
    Build spatial filtering frame
    '''
    def initialize_spatial_filtering_frame(self, spatial_filtering_frame):
        spatial_filtering_frame.place(x=0, y=self.functionality_frame_height * 3, width=self.functionality_frame_width, height=self.functionality_frame_height)
        # Initialize algorithms drop down menu and resize button
        self.spatial_filter_choice = StringVar(spatial_filtering_frame)
        self.spatial_filter_choice.set("Smoothing") # default value
        filters = ['Smoothing', 'Median', 'Sharpening Laplcian', 'High Boosting']
        self.filter_option_menu = OptionMenu(spatial_filtering_frame, self.spatial_filter_choice, filters[0], filters[1], filters[2], filters[3],
            command=self.spatial_fitering_option_menu_handler)
        self.filter_option_menu.pack(side=LEFT)

        width_label = Label(spatial_filtering_frame, text="width").pack(side=LEFT)
        self.spatial_filtering_mask_width = Text(spatial_filtering_frame, width=3, height=1, highlightbackground='black', highlightthickness=1)
        self.spatial_filtering_mask_width.pack(side=LEFT)
        self.spatial_filtering_mask_width.insert(END, 3)

        height_label = Label(spatial_filtering_frame, text="height").pack(side=LEFT)
        self.spatial_filtering_mask_height = Text(spatial_filtering_frame, width=3, height=1, highlightbackground='black', highlightthickness=1)
        self.spatial_filtering_mask_height.pack(side=LEFT)
        self.spatial_filtering_mask_height.insert(END, 3)

        k_label = Label(spatial_filtering_frame, text="K").pack(side=LEFT)
        self.high_boosting_filter_a = Text(spatial_filtering_frame, width=3, height=1, highlightbackground='black', highlightthickness=1, background="gray")
        self.high_boosting_filter_a.pack(side=LEFT)
        self.high_boosting_filter_a.insert(END, 3)
        self.high_boosting_filter_a.config(state=DISABLED) # Initalially disable if high boosting is not chosen

        spatial_filtering_button = Button(spatial_filtering_frame, text="Filtering", command=self.spatial_filtering)
        spatial_filtering_button.pack(side=LEFT)

    def spatial_fitering_option_menu_handler(self, *args):
        if self.spatial_filter_choice.get() == 'High Boosting':
            self.high_boosting_filter_a.config(background='white')
            self.high_boosting_filter_a.config(state=NORMAL)
        else:
            self.high_boosting_filter_a.config(background='gray')
            self.high_boosting_filter_a.config(state=DISABLED)
    
    def spatial_filtering(self):
        if self.spatial_filter_choice.get() == "Smoothing":
            new_img_array = img_processor.smoothing_filtering(self.img_array,
                int(self.spatial_filtering_mask_width.get('1.0', END)), int(self.spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)
        elif self.spatial_filter_choice.get() == "Median":
            new_img_array = img_processor.median_filtering(self.img_array,
                int(self.spatial_filtering_mask_width.get('1.0', END)), int(self.spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)
        elif self.spatial_filter_choice.get() == "Sharpening Laplcian":
            new_img_array = img_processor.sharpening_laplacian_filtering(self.img_array,
                int(self.spatial_filtering_mask_width.get('1.0', END)), int(self.spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)
        elif self.spatial_filter_choice.get() == "High Boosting":
            new_img_array = img_processor.high_boosting_filtering(self.img_array,
                int(self.spatial_filtering_mask_width.get('1.0', END)), int(self.spatial_filtering_mask_height.get('1.0', END)), float(self.high_boosting_filter_a.get('1.0', END)))
            self.update_image(new_img_array)

    '''
    Build bit panel frame
    '''
    def initialize_bit_panel_frame(self, bit_panel_frame):
        bit_panel_frame.place(x=0, y=self.functionality_frame_height * 4, width=self.functionality_frame_width, height=self.functionality_frame_height)
        bits_label = [0, 1, 2, 3, 4, 5, 6, 7]
        self.bits_vars = []
        for bit in bits_label:
            var = IntVar()
            var.set(1)
            check_button = Checkbutton(bit_panel_frame, text=bit, variable=var)
            check_button.pack(side=LEFT)
            self.bits_vars.append(var)
        resize_button = Button(bit_panel_frame, text="Update Bit Panel", command=self.update_bit_panel)
        resize_button.pack(side=LEFT)
    
    def update_bit_panel(self):
        bit_mask = 0
        for i in range(len(self.bits_vars)):
            bit_mask <<= 1
            if self.bits_vars[7 - i].get() == 1:
                bit_mask |= 1
        print(f"bit plane value: {bit_mask}")
        new_img_array = img_processor.update_bit_panel(self.img_array, bit_mask)
        self.update_image(new_img_array)

    '''
    Build noise generating frame
    '''
    def initialize_noise_generating_frame(self, noise_generating_frame):
        noise_generating_frame.place(x=0, y=self.functionality_frame_height * 5, width=self.functionality_frame_width, height=self.functionality_frame_height)
        # Initialize algorithms drop down menu and resize button
        self.noise_generating_choice = StringVar(noise_generating_frame)
        self.noise_generating_choice.set("Gaussian") # default value
        generators = ['Gaussian', 'Poisson', 'Salt and Pepper', 'Speckle']
        self.noise_generating_option_menu = OptionMenu(noise_generating_frame, self.noise_generating_choice, 
            generators[0], generators[1], generators[2], generators[3])
        self.noise_generating_option_menu.pack(side=LEFT)

        noise_generating_button = Button(noise_generating_frame, text="Generating Noise", command=self.noise_generating)
        noise_generating_button.pack(side=LEFT)
    
    def noise_generating(self):
        print("TODO: noice generating")
        if self.noise_generating_choice.get() == "Gaussian":
            new_img_array = img_processor.gaussian(self.img_array)
            self.update_image(new_img_array)
        elif self.noise_generating_choice.get() == "Poisson":
            new_img_array = img_processor.poisson(self.img_array)
            self.update_image(new_img_array)
        elif self.noise_generating_choice.get() == "Salt and Pepper":
            new_img_array = img_processor.salt_and_pepper(self.img_array)
            self.update_image(new_img_array)
        elif self.noise_generating_choice.get() == "Speckle":
            new_img_array = img_processor.speckle(self.img_array)
            self.update_image(new_img_array)

    '''
    Build restoration spatial filtering frame
    '''
    def initialize_restoration_spatial_filtering_frame(self, restoration_spatial_filtering_frame):
        restoration_spatial_filtering_frame.place(x=0, y=self.functionality_frame_height * 6, width=self.functionality_frame_width, height=self.functionality_frame_height)
        # Initialize algorithms drop down menu and resize button
        self.restoration_spatial_filter_choice = StringVar(restoration_spatial_filtering_frame)
        self.restoration_spatial_filter_choice.set("Arithmetic mean filter") # default value
        filters = ['Arithmetic mean filter', 'Geometric mean filter', 'Harmonic mean filter', 
                    'Contraharmonic mean filter', 'Max filter', 'Min fliter', 
                    'Midpoint filter', 'Alpha-trimmed mean filter']
        self.restoration_filter_option_menu = OptionMenu(restoration_spatial_filtering_frame, self.restoration_spatial_filter_choice, 
            filters[0], filters[1], filters[2], filters[3], filters[4], filters[5], filters[6], filters[7],
            command=self.restoration_spatial_fitering_option_menu_handler)
        self.restoration_filter_option_menu.pack(side=LEFT)

        width_label = Label(restoration_spatial_filtering_frame, text="width").pack(side=LEFT)
        self.restoration_spatial_filtering_mask_width = Text(restoration_spatial_filtering_frame, width=3, height=1, highlightbackground='black', highlightthickness=1)
        self.restoration_spatial_filtering_mask_width.pack(side=LEFT)
        self.restoration_spatial_filtering_mask_width.insert(END, 3)

        height_label = Label(restoration_spatial_filtering_frame, text="height").pack(side=LEFT)
        self.restoration_spatial_filtering_mask_height = Text(restoration_spatial_filtering_frame, width=3, height=1, highlightbackground='black', highlightthickness=1)
        self.restoration_spatial_filtering_mask_height.pack(side=LEFT)
        self.restoration_spatial_filtering_mask_height.insert(END, 3)

        # k_label = Label(restoration_spatial_filtering_frame, text="K").pack(side=LEFT)
        # self.high_boosting_filter_a = Text(restoration_spatial_filtering_frame, width=3, height=1, highlightbackground='black', highlightthickness=1, background="gray")
        # self.high_boosting_filter_a.pack(side=LEFT)
        # self.high_boosting_filter_a.insert(END, 3)
        # self.high_boosting_filter_a.config(state=DISABLED) # Initalially disable if high boosting is not chosen

        restoration_spatial_filtering_button = Button(restoration_spatial_filtering_frame, text="Filtering", command=self.restoration_spatial_filtering)
        restoration_spatial_filtering_button.pack(side=LEFT)

    def restoration_spatial_fitering_option_menu_handler(self, *args):
        print("restoration_spatial_fitering_option_menu_handler(self)")

    def restoration_spatial_filtering(self):
        if self.restoration_spatial_filter_choice.get() == "Arithmetic mean filter":
            new_img_array = img_processor.arithmetic_mean_filtering(self.img_array,
                int(self.restoration_spatial_filtering_mask_width.get('1.0', END)), int(self.restoration_spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)
        elif self.restoration_spatial_filter_choice.get() == "Geometric mean filter":
            new_img_array = img_processor.geometric_mean_filtering(self.img_array,
                int(self.restoration_spatial_filtering_mask_width.get('1.0', END)), int(self.restoration_spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)
        elif self.restoration_spatial_filter_choice.get() == "Harmonic mean filter":
            new_img_array = img_processor.harmonic_mean_filtering(self.img_array,
                int(self.restoration_spatial_filtering_mask_width.get('1.0', END)), int(self.restoration_spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)
        elif self.restoration_spatial_filter_choice.get() == "Contraharmonic mean filter":
            new_img_array = img_processor.contraharmonic_mean_filtering(self.img_array,
                int(self.restoration_spatial_filtering_mask_width.get('1.0', END)), int(self.restoration_spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)
        elif self.restoration_spatial_filter_choice.get() == "Max filter":
            new_img_array = img_processor.max_filtering(self.img_array,
                int(self.restoration_spatial_filtering_mask_width.get('1.0', END)), int(self.restoration_spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)
        elif self.restoration_spatial_filter_choice.get() == "Min filter":
            new_img_array = img_processor.min_filering(self.img_array,
                int(self.restoration_spatial_filtering_mask_width.get('1.0', END)), int(self.restoration_spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)
        elif self.restoration_spatial_filter_choice.get() == "Midpoint filter":
            new_img_array = img_processor.midpoint_filtering(self.img_array,
                int(self.restoration_spatial_filtering_mask_width.get('1.0', END)), int(self.restoration_spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)
        elif self.restoration_spatial_filter_choice.get() == "Alpha-trimmed mean filter":
            new_img_array = img_processor.alpha_trimmed_mean_filtering(self.img_array,
                int(self.restoration_spatial_filtering_mask_width.get('1.0', END)), int(self.restoration_spatial_filtering_mask_height.get('1.0', END)))
            self.update_image(new_img_array)

    ''' 
    Build image helper frame
    '''
    def initialize_image_helper_frame(self, image_helper_frame):
        image_helper_frame.pack(padx=20, pady=20)
        image_helper_frame.place(x=0, y=self.image_frame_height / 2, width=self.functionality_frame_width, height=self.functionality_frame_height)
        # Pop up original image
        popup_button = Button(image_helper_frame, text="Original Image", command=self.popup_original_image)
        popup_button.pack(side=LEFT)
        # Button for histogram displaying
        display_histogram_button = Button(image_helper_frame, text='Histogram Diagram', command=self.show_histogram)
        display_histogram_button.pack(padx=5, side=LEFT)
        # Button for saving modification
        modification_saving_button = Button(image_helper_frame, text='Save Modification', command=self.save_modification)
        modification_saving_button.pack(padx=5, side=LEFT)

    def popup_original_image(self): 
        popup_image_window = Toplevel()
        # Set information of the window
        ori_img_width, ori_img_height = self.ori_img.size
        popup_image_window.wm_title(f"{self.img_path.split('/')[-1]} ({ori_img_width} x {ori_img_height})")
        popup_image_label = Label(popup_image_window)
        popup_image_label.pack()
        # Display the image 
        display_img = ImageTk.PhotoImage(Image.open(self.img_path))
        popup_image_label.configure(image=display_img)
        popup_image_label.image = display_img

    def show_histogram(self):
        vals = self.modified_img_array.flatten()
        b, bins, patches = plt.hist(vals, 255)
        plt.xlim([0, 255])
        plt.show()
    
    def save_modification(self):
        self.img_array = self.modified_img_array.copy()
        self.new_width = len(self.img_array[0])
        self.new_height = len(self.img_array)

    '''
    Build zooming shrinking frame
    '''
    def initialize_zoom_shrink_frame(self, zoom_shrink_frame):
        zoom_shrink_frame.place(x=0, y=self.image_frame_height/2 + self.functionality_frame_height, width=self.functionality_frame_width, height=100)
        self.zoom_shrink_scale = Scale(zoom_shrink_frame, label='Zoom Shrink Scale', from_=0, to=5, orient=HORIZONTAL,
             length=400, showvalue=1, tickinterval=1, resolution=0.01, command=self.activate_zoom_shrink) 
        self.zoom_shrink_scale.set(1)
        self.zoom_shrink_scale.pack(side=LEFT)

    def activate_zoom_shrink(self, value):
        new_width = int(float(len(self.modified_img_array[0])) * float(value))
        new_height = int(float(len(self.modified_img_array)) * float(value))
        self.img_info.configure(text=f"{new_width} x {new_height}")
        self.modified_img = Image.fromarray(self.modified_img_array)
        self.modified_img = self.modified_img.resize((new_width, new_height))
        self.display_img = ImageTk.PhotoImage(self.modified_img)
        self.image_label.configure(image=self.display_img)

    '''
    Build image displaying frame
    '''
    def initialize_image_frame(self, image_frame):
        image_frame.pack(padx=10, pady=10)
        image_frame.place(x=self.functionality_frame_width, y=0, width=self.image_frame_width, height=self.image_frame_height)
        # Image Info
        self.img_info = Label(image_frame)
        self.img_info.pack(side=TOP)
        # Image Displaying Label
        self.image_label = Label(image_frame)
        self.image_label.pack()
        self.image_label.place(relx=.5, rely=.5, anchor="center")
        # load default image and display it
        self.img_array = numpy.array(Image.open(self.img_path))
        self.update_image(self.img_array.copy())
    
    def update_image(self, new_img_array):
        self.modified_img_array = new_img_array
        self.img_info.configure(text=f"{len(self.modified_img_array[0])} x {len(self.modified_img_array)}")
        self.display_img = ImageTk.PhotoImage(Image.fromarray(self.modified_img_array))
        self.image_label.configure(image=self.display_img)
        self.zoom_shrink_scale['variable'] = DoubleVar(value=1.0)

def main():
    root = Tk()
    app = Window(root)
    root.wm_title("Image Processor")
    root.mainloop()

if __name__ == '__main__':
    main()
