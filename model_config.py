from tkinter import *
from tkinter import ttk
import numpy as np

def display_paramaters(frm, numberOfLayer):
    n = numberOfLayer.get()
    if not n:
        return
    # faire une boucle pour chaque layer et mettre les options dans un tableau numpy
    for i in range(n):
        pass

def main():
    try :
        root = Tk()
        root.title("Interface Propre Tkinter")
        root.geometry("800x400")
        root.resizable(False, False)

        # Créer un frame principal
        frm = ttk.Frame(root, padding=20)
        frm.grid(sticky="nsew")  # Le frame remplit la fenêtre

        for i in range(5):  # Exemple avec 5 lignes
            frm.grid_rowconfigure(i, minsize=40)  # Taille par défaut pour chaque ligne

        for i in range(5):  # Exemple avec 3 colonnes
            frm.grid_columnconfigure(i, minsize=100)

        ttk.Label(frm, text='Select number of hidden layer').grid(column=0, row=0)
        numberOfLayer = ttk.Combobox(frm, values=(2, 3, 4, 5))
        numberOfLayer.grid(column=0,row=1)

        ttk.Button(frm, text='ok', command=lambda : display_paramaters(frm, numberOfLayer)).grid(column=1, row=1)

        entry = ttk.Entry(frm, textvariable='blabla')
        entry.grid(column=0, row=5)

        ttk.Button(frm, text="Quit", command=root.destroy).grid(column=10, row=10)
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__== "__main__":
    main()