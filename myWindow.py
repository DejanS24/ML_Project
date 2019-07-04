import tkinter as tk
from tkinter import ttk
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
import linearregression
import numpy as np
import pandas as pd
import linearregression
from svm import svm_prediction
from main import load_file, knn_predict, train_valid_split, preprocess, plot_graph, auto_arima_predict,prophet_predict
import matplotlib.pyplot as plt

class MyWindow:
    def __init__(self):
        self.window = tk.Tk()
        self.window.wm_title("Stock prediction using ML")
        self.window.config(background="#FFFFFF")
        self.cap = None
        self.mode = "Google"
        self.algorithm = ""
        self.file_path = ""

        # self.video_thread = Thread(target=self.display_video)

        self.imageFrame = tk.Frame(self.window, width=640, height=580)
        self.imageFrame.grid(row=0, column=0, padx=10, pady=2)

        self.company_label = tk.Label(self.imageFrame, text="Company:")
        self.company_label.grid(row=0, column=0, sticky=tk.W, pady=2, padx=4)

        self.combobox_types = ttk.Combobox(self.imageFrame,values=["Google", "Apple", "Amazon", "Coca Cola"], state="readonly")
        self.combobox_types.set("Google")
        self.combobox_types.grid(row=0, column=1, sticky=tk.W, pady=2, padx=4)

        self.algorithm_label = tk.Label(self.imageFrame, text="Algorithm:")
        self.algorithm_label.grid(row=0, column=2, sticky=tk.W, pady=2, padx=4)

        self.combobox_algs = ttk.Combobox(self.imageFrame, values=["Linear Regression", "SVM", "Moving average", "Auto Arima", "KNN"], state="readonly")
        self.combobox_algs.set("KNN")
        self.combobox_algs.grid(row=0, column=3, sticky=tk.W, pady=2, padx=4)

        self.ldBtn = tk.Button(self.imageFrame, text="Load", command=self.display_graph, width=5)
        self.ldBtn.grid(row=0, column=3, sticky=tk.E, pady=2, padx=4)

        self.main_frame = tk.Label(self.imageFrame)
        self.main_frame.grid(row=1, column=0, columnspan=4, sticky=tk.S)

        # self.faceDetector = FaceDetector()
        # self.ec = EmotionClassifier()
        # self.go = GenderClassifier()
        # self.video_thread.start()
        self.window.mainloop()

    def display_graph(self):
        self.algorithm = self.combobox_algs.get()
        self.mode = self.combobox_types.get()
        print(self.mode)
        print(self.algorithm)


        if self.mode == "Google":
            self.file_path = "googl.us.txt"
        elif self.mode == "Apple":
            self.file_path = "aapl.us.txt"
        elif self.mode == "Amazon":
            self.file_path = "amzn.us.txt"
        elif self.mode == "Coca Cola":
            self.file_path = "ko.us.txt"


        # df = load_file(self.file_path)

        df = pd.read_csv(self.file_path)

        if self.algorithm == "SVM":
            print("Svm izabran")
            # self.y_train, self.y_val, self.y_predict = svm_prediction(df)

            new_df = preprocess(load_file(self.file_path))
            x_train, y_train, x_valid, y_valid, self.train, self.valid = train_valid_split(new_df)
            self.y_predict = svm_prediction(df, x_train, y_train, x_valid, y_valid)
            self.y_train = y_train
            self.y_val = y_valid



        elif self.algorithm == "Moving average":
            print("MA izabran")
            #NISAM TESTIRAO MA, TREBA MODIFIKOVATI

        elif self.algorithm == "KNN":
            print("KNN izabran")
            df = load_file(self.file_path)
            new_df = preprocess(df)
            x_train, y_train, x_valid, y_valid, self.train, self.valid = train_valid_split(new_df)
            self.y_predict = knn_predict(x_train, y_train, x_valid)
            self.y_train = y_train
            self.y_val = y_valid

            plt.plot(self.y_train)
            plt.plot(self.y_val)
            plt.plot(self.y_predict)
            plt.show()

            # plot_graph(self.train, self.valid, self.y_predict)
        elif self.algorithm == "Auto Arima":
            print("Auto Arima izabran")
            # UBACI OVDE POZIV AUTO ARIMA METODE
            df = load_file(self.file_path)
            new_df = preprocess(df)
            x_train, y_train, x_valid, y_valid, self.train, self.valid = train_valid_split(new_df)
            self.y_predict = auto_arima_predict(df)
            self.y_train = y_train
            self.y_val = y_valid

        elif self.algorithm == "Linear Regression":
            print("Linear Regression izabran")
            # UBACI OVDE POZIV LINEAR REGRESSION
            df = load_file(self.file_path)
            new_df = preprocess(df)
            x_train, y_train, x_valid, y_valid, self.train, self.valid = train_valid_split(new_df)
            self.y_predict = linearregression.run_regression(x_train, y_train, x_valid, y_valid)
            self.y_train = y_train
            self.y_val = y_valid

            plt.plot(self.y_train)
            plt.plot(self.y_val)
            plt.plot(self.y_predict)
            plt.show()
        elif self.algorithm == "Prophet":
            print("Prophet")
            df = load_file(self.file_path)
            new_df = preprocess(df)
            x_train, y_train, x_valid, y_valid, self.train, self.valid = train_valid_split(new_df)
            self.y_predict = prophet_predict(df, self.valid)
            self.y_train = y_train
            self.y_val = y_valid

            plt.plot(self.y_train)
            plt.plot(self.y_val)
            plt.plot(self.y_predict)
            plt.show()


        print(self.y_predict)

        print("Zavrsio obucavanje i predikciju")


        p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")
        p1.grid.grid_line_alpha = 0.3
        p1.xaxis.axis_label = 'Date'
        p1.yaxis.axis_label = 'Price'

        plot_dates = df['Date']
        print(self.valid)
        plot_dates = plot_dates[-len(self.y_predict):]
        p1.line(plot_dates, self.valid['Close'], color='#A6CEE3', legend=self.mode)
        p1.line(plot_dates, self.y_predict, color='#B2DF8A', legend="Predicted "+self.mode)

        output_file("stocks.html", title="Stocks prediction")

        show(gridplot([[p1]], plot_width=500, plot_height=500))

if __name__ == "__main__":

    winddow = MyWindow()
