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
    self.boundary_condition.items = ["Clamped", "Simply supported"]

    # Initialize plate
    self.platefigure_reset()
    self.canvas_progress_reset(0)

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

  
  def convert_boundary_condition_value(self, condition):
    if condition == "Clamped":
      return "f"
    elif condition == "Simply supported":
      return "s"

  def Input_click(self, **event_args):
    """This method is called when the button is clicked"""
    boundary_condition = self.convert_boundary_condition_value(self.boundary_condition.selected_value)
    self.E = self.input_E.text if self.input_E.text else "206e09"
    self.mu = self.input_mu.text if self.input_mu.text else "0.3"
    self.W = self.input_W.text if self.input_W.text else "2"
    self.H = self.input_H.text if self.input_H.text else "2"
    self.t = self.input_t.text if self.input_t.text else "0.001"
    self.q = self.input_q.text if self.input_q.text else "0"
    self.lr = self.input_lr.text if self.input_lr.text else "0.01"
    self.epochs = self.input_epochs.text if self.input_epochs.text else "210"

    # 백그라운드 태스크 시작
    task_id = anvil.server.call(
            'launch_calculate_plate', boundary_condition,self.E,self.mu,self.W,self.H,self.t,self.q,self.lr, int(self.epochs)
        )
    
    # 진행 상황 폴링
    while True:
        progress_data = anvil.server.call('get_task_progress', task_id)
        progress = progress_data.get('progress', 0)
        self.canvas_progress_reset(progress)
      
        if not progress_data['running']:
          result_text = progress_data['result_text']
          break

    
    image_3d=anvil.server.call('create_image',"/tmp/plate_3d_plot.png")
    image_2d=anvil.server.call('create_image',"/tmp/plate_2d_plot.png")
    self.image_plate_deflection.source = image_3d
    self.image_plate_deflection.width = "1000px"
    self.image_plate_deflection.height = "800px"

    self.image_plate_displacement.source = image_2d
    self.image_plate_displacement.width = "990px"
    self.image_plate_displacement.height = "400px"
    self.text_result.text = result_text
    self.text_result.height = "110px"

  def canvas_progress_reset(self, progress=0,**event_args):
    """This method is called when the canvas is reset and cleared, such as when the window resizes, or the canvas is added to a form."""
    canvas = self.canvas_progress
    canvas.clear_rect(0, 0, canvas.get_width(), canvas.get_height())
    canvas.begin_path()
    canvas.fill_style = "#e0e0e0"
    canvas.fill_rect(10, self.canvas_progress.get_height() - 30, self.canvas_progress.get_width() - 20, 20)
    canvas.fill_style = "#76c7c0"
    canvas.fill_rect(10, self.canvas_progress.get_height() - 30, (self.canvas_progress.get_width() - 20) * (progress /float(self.input_epochs.text) ), 20)
    canvas.fill_style = "#000000"
    canvas.font = "16px Arial"
    canvas.fill_text(f"{int(progress)}/{self.input_epochs.text}", self.canvas_progress.get_width() / 2 - 10, self.canvas_progress.get_height() - 35)



  



