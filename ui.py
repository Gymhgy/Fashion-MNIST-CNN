from tkinter import Tk, filedialog
from tkinter.ttk import Frame, Button, Label
import torch
from PIL import ImageTk, Image
import torchvision.transforms.functional as TF

results = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

def resize(image):
    return ImageTk.PhotoImage(Image.open(image).resize((140,140), Image.ANTIALIAS))

class MainFrame(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        try:
            self.model = torch.load("model.pt")
        except:
            pass
        self.imgframe = Frame(self, height=140, width=140)
        self.img = None
        self.image = Label(self.imgframe)
        self.image.grid()
        
        self.result = Label(self, text="Model thinks it is: ", anchor="w")
        
        def upload():
            file = filedialog.askopenfilename(initialdir = "samples")
            self.img = resize(file)
            self.image.config(image = self.img)
            pred = self.model(TF.to_tensor(Image.open(file)).unsqueeze(0)).argmax().item()
            self.result.config(text = "Model thinks it is: " + results[pred])
            pass
        self.upload_button = Button(self, text = "Upload", command=upload)
        self.imgframe.grid_propagate(0)
        self.imgframe.grid(row=0,column=0)
        self.upload_button.grid(row=0,column=1)
        self.result.grid(row=1, column=0, columnspan=2)


if __name__ == "__main__":
    root = Tk()
    root.geometry("220x160")
    root.resizable(False, False)

    root.title("Fashion MNIST CNN")
    MainFrame(root).pack(side="bottom", fill="both", expand=True)
    root.mainloop()