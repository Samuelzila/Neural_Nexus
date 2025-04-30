from denserflow import models
import interface

# --- Load the interface --- #
# Load the model
model1 = models.load_model("models/NeuralNexus0,8850528270419435.json")
model1.name = "NeuralNexus0,8850528270419435"
model2 = models.load_model("models/NeuralNexus0,884915279.json")
model2.name = "NeuralNexus0,884915279"
model3 = models.load_model("models/NeuralNexus0,884889.json")
model3.name = "NeuralNexus0,884889"
model4 = models.load_model("models/NeuralNexus0,88.json")
model4.name = "NeuralNexus0,88"
model5 = models.load_model("models/NeuralNexus0,87.json")
model5.name = "NeuralNexus0,87"
model6 = models.load_model("models/always_right.json")
model6.name = "always_right"

modeles = [model1, model2, model3, model4, model5, model6]

# Run the tkinter app
app = interface.DrawingApp(interface.canvas_frame, modeles)
interface.set_app_instance(app)  # <-- nouvelle fonction à définir dans interface.py
app.pack()


interface.fenetre.mainloop()
