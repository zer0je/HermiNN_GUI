from ._anvil_designer import plateTemplate
from anvil import *
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import plotly.graph_objects as go
import anvil.server


class plate(plateTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Set Color
    self.Input.background = "#CED8F6"

    # Initialize dropdown menus
    self.boundary_condition.items = ["Clamped","Simply supported"]

    # Initialize plate
    self.platefigure_reset()

  def platefigure_reset(self, **event_args):
    """This method is called when the canvas is reset and cleared, such as when the window resizes, or the canvas is added to a form."""
    self.create_plate()


  def create_plate(self):
    """This method is called when the canvas is reset and cleared, such as when the window resizes, or the canvas is added to a form."""
    canvas = self.platefigure
    width, height = canvas.get_width(), canvas.get_height()

    # Clear the canvas by drawing a white rectangle over the entire canvas
    canvas.fill_style = "#FFFFFF"  # White color
    canvas.fill_rect(0, 0, width, height)

    # Draw a square plate
    plate_width = 150 
    plate_height = 150 

    # Calculate positions
    x_start = (width - plate_width) / 2
    y_start = (height - plate_height) / 2

    # Set drawing style
    canvas.fill_style = "#000000"  # Black color
    canvas.fill_rect(x_start, y_start, plate_width, plate_height)

    # Draw x-axis label
    canvas.fill_style = "#000000"
    canvas.font = "16px Arial"
    canvas.fill_text('height', x_start - 80, y_start + plate_height / 2 + 5)  
    
    # Draw y-axis label
    canvas.fill_text('width', x_start + plate_width / 2 - 10, y_start - 5)




  def convert_boundary_condition_value(self, condition):
    if condition == "Clamped":
      return 'f' 
    elif condition == "Simply supported":
      return 's'

  def Input_click(self, **event_args):
    """This method is called when the button is clicked"""
    boundary_condition=self.convert_boundary_condition_value(self.boundary_condition.selected_value)
    self.E = self.input_E.text if self.input_E.text else "206e09"
    self.mu= self.input_mu.text if self.input_mu.text else "0.3"
    self.width = self.input_width.text if self.input_width.text else "2"
    self.height = self.input_height.text if self.input_height.text else "2"
    self.h= self.input_h.text if self.input_h.text else "0.001"
    self.q = self.input_q.text if self.input_q.text else "0"
    self.lr = self.input_lr.text if self.input_lr.text else "0.01"
    self.epochs = self.input_epochs.text if self.input_epochs.text else "210"

    anvil.server.call("initialize_plate_parameters",boundary_condition,self.E,self.mu,self.width,self.height,self.h,self.q,self.lr,self.epochs)
    img_3d,img_2d, result = anvil.server.call("calculate_plate")
    self.image_plate_deflection.source = img_3d
    self.image_plate_deflection.width = "1000px"
    self.image_plate_deflection.height = "800px"
    self.image_plate_deflection.horizontal_align='center'
    self.image_plate_deflection.spacing_above='small'
    self.image_plate_deflection.spacing_below='small'
    
    self.image_plate_displacement.source = img_2d
    self.image_plate_displacement.width = "1000px"
    self.image_plate_displacement.height = "800px"
    self.image_plate_displacement.spacing_above='small'
    self.image_plate_displacement.spacing_below='small'
    
    self.text_result.text = result
    self.text_result.height = "110px"

