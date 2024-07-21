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
    self.left_boundary_condition.items = ["Free", "Simply supported", "Fixed"]
    self.right_boundary_condition.items = ["Free", "Simply supported", "Fixed"]

    # Attach change event handlers
    self.left_boundary_condition.set_event_handler(
      "change", self.left_boundary_condition_change
    )
    self.right_boundary_condition.set_event_handler(
      "change", self.right_boundary_condition_change
    )

    # Iniate canvas drawing
    self.beamfigure_reset()

  def plate_figure_reset(self, **event_args):
    """This method is called when the canvas is reset and cleared, such as when the window resizes, or the canvas is added to a form."""
    x_start, y_start, beam_length, beam_height = self.create_beam()

  def create_beam(self):
    """This method is called when the canvas is reset and cleared, such as when the window resizes, or the canvas is added to a form."""
    canvas = self.beamfigure
    width, height = canvas.get_width(), canvas.get_height()

    # Clear the canvas by drawing a white rectangle over the entire canvas
    canvas.fill_style = "#FFFFFF"  # White color
    canvas.fill_rect(0, 0, width, height)

    # Draw a horizontal beam (1D beam)
    beam_length = width * 0.8  # 80% of the canvas width
    beam_height = 10  # Fixed height for the beam

    # Calculate positions
    x_start = (width - beam_length) / 2
    y_start = height / 2 - beam_height / 2

    # Set drawing style
    canvas.fill_style = "#000000"  # Black color
    canvas.fill_rect(x_start, y_start, beam_length, beam_height)

    # Draw arrow indicating direction (rightward arrow at the end of the beam)
    arrow_length = 80
    arrow_height = 20
    x_arrow_start = x_start - arrow_length
    y_arrow_start = y_start + beam_height - 45

    canvas.stroke_style = "#000000"
    canvas.begin_path()
    canvas.move_to(x_arrow_start, y_arrow_start)
    canvas.line_to(x_arrow_start + arrow_length, y_arrow_start)
    canvas.line_to(
      x_arrow_start + arrow_length - arrow_height / 2, y_arrow_start - arrow_height / 2
    )
    canvas.move_to(x_arrow_start + arrow_length, y_arrow_start)
    canvas.line_to(
      x_arrow_start + arrow_length - arrow_height / 2, y_arrow_start + arrow_height / 2
    )
    canvas.stroke()

    # Draw x-axis label
    canvas.fill_style = "#000000"
    canvas.font = "14px Arial"
    canvas.fill_text("x", x_arrow_start + arrow_length + 10, y_arrow_start + 5)

    # Return beam position and dimensions
    return x_start, y_start, beam_length, beam_height

  def draw_boundary_conditions(self, x_start, y_start, beam_length, beam_height):
    conditions = {
      "Free": self.draw_free,
      "Simply supported": self.draw_simply_supported,
      "Fixed": self.draw_fixed,
    }

    # Draw left boundary condition
    left_condition = self.left_boundary_condition.selected_value
    conditions[left_condition](x_start, y_start, beam_height)

    # Draw right boundary condition
    right_condition = self.right_boundary_condition.selected_value
    conditions[right_condition](x_start + beam_length, y_start, beam_height)

  def draw_free(self, x, y, beam_height):
    # Free boundary condition has no specific drawing, just a placeholder
    pass

  def draw_simply_supported(self, x, y, beam_height):
    canvas = self.beamfigure
    canvas.fill_style = "#000000"
    # Draw a triangle for simply supported condition
    canvas.begin_path()
    canvas.move_to(x, y + beam_height)
    canvas.line_to(x - 15, y + beam_height + 30)
    canvas.line_to(x + 15, y + beam_height + 30)
    canvas.close_path()
    canvas.fill()

  def draw_fixed(self, x, y, beam_height):
    canvas = self.beamfigure
    canvas.fill_style = "#000000"

    # Draw a thicker vertical rectangle for fixed condition extending above and below the beam
    wall_thickness = 20  # Thickness of the wall
    wall_height = beam_height + 40  # Total height of the wall

    # Draw the wall above and below the beam
    canvas.fill_rect(x - wall_thickness / 2, y - 20, wall_thickness, wall_height)

  def left_boundary_condition_change(self, **event_args):
    """This method is called when the selected value of this drop down changes"""
    x_start, y_start, beam_length, beam_height = self.create_beam()
    self.draw_boundary_conditions(x_start, y_start, beam_length, beam_height)

  def right_boundary_condition_change(self, **event_args):
    """This method is called when the selected value of this drop down changes"""
    x_start, y_start, beam_length, beam_height = self.create_beam()
    self.draw_boundary_conditions(x_start, y_start, beam_length, beam_height)

  def convert_boundary_condition_value(self, condition):
    if condition == "Free":
      return "x"
    elif condition == "Simply supported":
      return "s"
    elif condition == "Fixed":
      return "f"

  def Input_click(self, **event_args):
    """This method is called when the button is clicked"""
    left_condition = self.convert_boundary_condition_value(
      self.left_boundary_condition.selected_value
    )
    right_condition = self.convert_boundary_condition_value(
      self.right_boundary_condition.selected_value
    )
    self.E = self.input_E.text if self.input_E.text else "1"
    self.I = self.input_I.text if self.input_I.text else "1"
    self.L = self.input_L.text if self.input_L.text else "1"
    self.P = self.input_P.text if self.input_P.text else "0"
    self.x_p = self.input_x_p.text if self.input_x_p.text else "0"
    self.q = self.input_q.text if self.input_q.text else "0"
    self.lr = self.input_lr.text if self.input_lr.text else "0.2"
    self.epochs = self.input_epochs.text if self.input_epochs.text else "200"

    anvil.server.call(
      "initialize_beam_parameters",
      left_condition,
      right_condition,
      self.E,
      self.I,
      self.L,
      self.P,
      self.x_p,
      self.q,
      self.lr,
      self.epochs,
    )
    img_media, result = anvil.server.call("calculate_beam")
    self.image_beam_deflection.source = img_media
    self.image_beam_deflection.width = "1000px"
    self.image_beam_deflection.height = "800px"
    self.text_result.text = result
    self.text_result.height = "110px"
