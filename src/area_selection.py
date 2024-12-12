import os
import csv
import tkinter as tk
from PIL import Image, ImageTk, ImageGrab

class AreaSelector:
   def __init__(self, image_path, coords_file='outputs/coords.csv'):
      self.image_path = image_path
      self.canvas = None
      self.window = None
      self.width_ratio, self.height_ratio = 0, 0
      self.topx, self.topy, self.botx, self.boty = 0, 0, 0, 0
      self.rect_id = None
      self.rect_count = 0
      self.coords_file = coords_file
      if os.path.exists(self.coords_file):
         os.remove(self.coords_file)

   def run(self):
      # Create window and canvas
      self.window = tk.Tk()
      self.window.title("Select Area")
      self.window.configure(background='grey')

      # Load the image and get its dimensions
      image = Image.open(self.image_path)
      screen_width, screen_height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
      max_width, max_height = min(image.size[0], screen_width), min(image.size[1], screen_height)

      # Calculate the aspect ratio of the image
      aspect_ratio = image.size[0] / image.size[1]

      # Calculate the new dimensions of the image
      if aspect_ratio > 1:
         # Landscape orientation
         new_width = max_width
         new_height = int(new_width / aspect_ratio)
      else:
         # Portrait orientation
         new_height = max_height
         new_width = int(new_height * aspect_ratio)

      self.width_ratio = new_width/image.size[0]
      self.height_ratio = new_height/image.size[1]
      # Resize the image
      image = image.resize((new_width, new_height))

      # Create the canvas with the dimensions of the resized image
      self.canvas = tk.Canvas(self.window, width=new_width, height=new_height, borderwidth=0, highlightthickness=0)
      self.canvas.pack(expand=True)

      # Load the image onto the canvas
      img = ImageTk.PhotoImage(image)
      self.canvas.img = img  # Keep reference in case this code is put into a function.
      self.canvas.create_image(0, 0, image=img, anchor=tk.NW)

      # Bind mouse events
      self.canvas.bind('<Button-1>', self.get_mouse_posn)
      self.canvas.bind('<B1-Motion>', self.update_sel_rect)
      self.canvas.bind('<ButtonRelease-1>', self.print_sel_rect)
      self.canvas.bind('<Button-3>', self.deselect_area)

      # Run the main loop
      self.window.protocol("WM_DELETE_WINDOW", self.save_image)
      self.window.mainloop()

   def get_mouse_posn(self, event):
      self.topx, self.topy = event.x, event.y

   def update_sel_rect(self, event):
      self.botx, self.boty = event.x, event.y
      if self.rect_id:
         self.canvas.coords(self.rect_id, self.topx, self.topy, self.botx, self.boty)
         self.canvas.tag_raise(self.rect_id)  # Move rectangle to the top of the canvas
      else:
         self.rect_id = self.canvas.create_rectangle(self.topx, self.topy, self.botx, self.boty,
                                          dash=(2,2), fill='', outline='blue', width=2)
         self.canvas.tag_lower(self.rect_id)  # Move rectangle to the bottom of the canvas
      self.canvas.itemconfig(self.rect_id, tags=('roi', 'roi{}'.format(self.rect_id)))
      

   def print_sel_rect(self, event):
      print("Selected area: ({}, {}) - ({}, {})".format(self.topx, self.topy, self.botx, self.boty), end=', ')
      print("Image coord: ({}, {}) - ({}, {})".format(int(self.topx/self.width_ratio), int(self.topy/self.height_ratio),
                                         int(self.botx/self.width_ratio), int(self.boty/self.height_ratio)))
      x1, y1, x2, y2 = self.canvas.coords(self.rect_id)
      x1_img, y1_img, x2_img, y2_img = int(x1/self.width_ratio), int(y1/self.height_ratio), \
                                       int(x2/self.width_ratio), int(y2/self.height_ratio)
      margin = 20
      label_x = x1
      label_y = y1 - margin
      label = 'ROI {}'.format(self.rect_count+1)
      label_id = self.canvas.create_text(label_x, label_y, text=label,
                                          anchor=tk.NW, fill='blue', font='Helvetica 12 bold')
      self.canvas.itemconfig(label_id, tags=('label', 'label{}'.format(label_id)))
      self.canvas.tag_raise('label{}'.format(self.rect_id))  # Move label to the top of the canvas
      self.canvas.addtag_withtag('roi{}'.format(self.rect_id), label_id)
      self.save_coords(label, (x1_img, y1_img, x2_img, y2_img))
      self.rect_count += 1
      self.rect_id = None


   def save_coords(self, label, coords):
      with open(self.coords_file, 'a', newline='') as f:
         writer = csv.writer(f)
         if f.tell() == 0:
               writer.writerow([f'# Image file name: {self.image_path}'])
               writer.writerow(['ROI', 'x1', 'y1', 'x2', 'y2'])
         writer.writerow([label, *map(int, coords)])

   def deselect_area(self, event):
      item = self.canvas.find_withtag(tk.CURRENT)
      if 'label' in self.canvas.gettags(item):
         for tag in self.canvas.itemcget(item, 'tags').split():
            if tag.startswith('roi'):
               rect_item = self.canvas.find_withtag(tag)[0]
               label_item = self.canvas.find_withtag('label{}'.format(item[0]))
               label_text = self.canvas.itemcget(label_item, 'text')
               print("Deselected area: {}".format(label_text))
               self.canvas.delete(item)
               self.canvas.delete(rect_item)
               # self.rect_count -= 1
               self.update_coords_file(label_text)


   def update_coords_file(self, deleted_label):
      """
      Update the coordinates file by removing the row with the specified label.

      Args:
         deleted_label (str): The label of the row to be deleted.
      """
      with open(self.coords_file, 'r', newline='', encoding='utf-8') as file:
         reader = csv.reader(file)
         rows = list(reader)
      with open(self.coords_file, 'w', newline='', encoding='utf-8') as file:
         writer = csv.writer(file)
         writer.writerow(rows[0])
         writer.writerow(rows[1])
         for row in rows[2:]:
            if row[0] != deleted_label:
               writer.writerow(row)

   def save_image(self):
      # Save the canvas to a file
      # x_pos = self.canvas.winfo_rootx()
      # y_pos = self.canvas.winfo_rooty()
      # width = self.canvas.winfo_width()
      # height = self.canvas.winfo_height()
      # canvas_image = ImageGrab.grab(bbox=(x_pos, y_pos, width, height))
      # canvas_image.save(os.path.splitext(self.image_path)[0] + '_roi.jpg', 'JPEG')
      filename = os.path.splitext(self.image_path)[0] + '_roi'
      self.canvas.postscript(file=filename+'.eps', colormode='color')
      
      # Close the window
      self.window.destroy()


if __name__ == '__main__':
   image_path = '/workspaces/MeteorDection/test/merged_rectified.jpg'
   coords_file = 'outputs/coords.csv'

   # Create the AreaSelector object and run it
   selector = AreaSelector(image_path, coords_file)
   selector.run()
