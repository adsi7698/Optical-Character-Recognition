#import modules
import Tkinter
from Tkinter import *
import os
import run_training
import run_classify

# Designing window for prediction 
def predict():
    global predict_screen
    print(predict_screen)
    predict_screen = Toplevel(main_screen)
    predict_screen.title("Predict The Character")
    predict_screen.geometry("300x300")

    Label(predict_screen, text="Ready for Prediction...", bg="blue", width="300", height="2", font=("Calibri", 13)).pack()
    Button(predict_screen, text="Predict", height="3", width="30", command=prediction).pack()

def prediction():
	s = run_classify.run_class()
	print(s)
	character, accuracy = s.split(',')

	Label(predict_screen, text=character+" is the predicted output", width="300", height="2", font=("Calibri", 13)).pack()
	Label(predict_screen, text=accuracy+" is the accuracy", width="300", height="2", font=("Calibri", 13)).pack()
 
 
# Executing run_training module using train()
def train():
	run_training.run_train()
 
def main_account_screen():
    global main_screen

    main_screen = Tk()
    main_screen.geometry("300x300")
    main_screen.title("Optical Character Recognition")

    Label(text="Select Your Choice", bg="blue", width="300", height="2", font=("Calibri", 13)).pack()
    Label(text="").pack()
    
    Button(text="Train", height="2", width="30", command = train).pack()
    Label(text="").pack()
    Button(text="Prediction", height="2", width="30", command = predict).pack()
 
    main_screen.mainloop()
 
 
main_account_screen()