from denserflow import models
import interface

# --- Load the interface --- #
# Load the model
model = models.load_model("models/NeuralNexus0,8850528270419435.json")

# Run the tkinter app
app = interface.DrawingApp(interface.canvas_frame, model)
app.pack()

interface.fenetre.mainloop()
