from denserflow import models
import interface


# --- Load the interface --- #
model = models.load_model("NeuralNexus0,87.json")
app = interface.DrawingApp(interface.canvas_frame, model)
app.pack()

interface.fenetre.mainloop()
