from tkinter import *
from tkinter import ttk
import pandas as pd
import json

activationFunctions = ['relu', 'sigmoid', 'tanh', 'softmax', 'linear']
weightInitializers = ['uniform_init', 'glorot_uniform', 'he_normal', 'lecun_uniform']

df = pd.DataFrame(columns=['nbOfNeurons', 'weightInit', 'activationFunction'])
labels = []

def clear_array():
    global df
    global labels
    for _, row in df.iterrows():
        for feature in row:
            if feature is not None:
                feature.destroy()
    for la in labels:
        la.destroy()
    labels.clear()
    df = pd.DataFrame(columns=['nbOfNeurons', 'weightInit', 'activationFunction'])

def display_label(frm, str, column, row):
    global labels
    label = ttk.Label(frm, text=str)
    label.grid(column=column, row=row)
    labels.append(label)

def save_as_json(frm):
    global df
    if df.empty:
        display_label(frm, 'ERROR : Select nb of layers', 4, 1)
        return
    
    result = pd.DataFrame(columns=['nbOfNeurons', 'weightInit', 'activationFunction'])
    for idx, row in df.iterrows():
        if any(feat.get() is None or feat.get() == "" for feat in row):
            display_label(frm, 'ERROR : Empty field', 4, 1)
            return
        result.loc[idx] = [feat.get() for feat in row]
    
    for neurons in result['nbOfNeurons']:
        try:
            if not (isinstance(int(neurons), int) and 1 <= int(neurons) <= 100):
                display_label(frm, 'ERROR : Invalid numbers of neurons', 4, 1)
                return
        except ValueError:
            display_label(frm, 'ERROR : Invalid numbers of neurons', 4, 1)
            return
    result = result.T
    jsonDict = result.to_dict()
    with open("generated_config.json", "w") as file:
        json.dump(jsonDict, file, indent=4)


def display_paramaters(frm, numberOfLayer):
    global df
    global labels
    if numberOfLayer is None:
        return
    n = numberOfLayer.get()
    if not n:
        return
    n = int(n)
    clear_array()
    
    for i in range(n):
        # Layer name
        layerName = 'Layer', i + 1
        display_label(frm, ('Layer', i + 1), i, 3)
        
        # nb neurons
        display_label(frm, 'Number of neurons', i, 4)
        nbOfNeurons = ttk.Entry(frm)
        nbOfNeurons.grid(column=i, row=5)
        
        # Weights
        display_label(frm, 'Init weight settings', i, 6)
        weightInit = ttk.Combobox(frm, values=weightInitializers)
        weightInit.grid(column=i, row=7, padx=10, pady=10)
        
        # Activation function
        display_label(frm, 'Activation function', i, 8)
        activationFct = ttk.Combobox(frm, values=activationFunctions)
        activationFct.grid(column=i, row=9)
        
        df.loc[i] = [nbOfNeurons, weightInit, activationFct]

def main():
    try:
        root = Tk()
        root.title("Interface Propre Tkinter")
        root.geometry("1000x400")
        root.resizable(False, False)

        frm = ttk.Frame(root, padding=10)
        frm.grid(sticky="nsew")

        for i in range(10):
            frm.grid_rowconfigure(i, minsize=30)
            frm.grid_columnconfigure(i, minsize=100)

        ttk.Label(frm, text='Select number of hidden layer').grid(column=0, row=0)
        numberOfLayer = ttk.Combobox(frm, values=(2, 3, 4, 5))
        numberOfLayer.grid(column=0, row=1)

        ttk.Button(frm, text='ok', command=lambda: display_paramaters(frm, numberOfLayer)).grid(column=1, row=1)
        ttk.Button(frm, text='generate config', command=lambda: save_as_json(frm)).grid(column=3, row=1)

        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()