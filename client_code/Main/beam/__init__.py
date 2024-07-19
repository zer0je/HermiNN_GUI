from ._anvil_designer import beamTemplate
from anvil import *
import anvil.server


class beam(beamTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Set Color
    self.Input.background = "#A4A4A4"
    self.Input.foreground = "#000000"

    # Initialize dropdown menus
    self.left_boundary_condition.items = ["Free", "Simply supported", "Fixed"]
    self.right_boundary_condition.items = ["Free", "Simply supported", "Fixed"]

    # Attach change event handlers
    self.left_boundary_condition.set_event_handler('change',self.left_boundary_condition_change)
    self.right_boundary_condition.set_event_handler('change',self.right_boundary_condition_change)

    # Initial canvas drawing
    self.canvas_1_reset()

    # Initial Analysis Data
    self.E=1
    self.I=1
    self.L=1
    self.P=0
    self.x_p=0
    self.q=0


  def canvas_1_reset(self, **event_args):
        """This method is called when the canvas is reset and cleared, such as when the window resizes, or the canvas is added to a form."""
        x_start, y_start, beam_length, beam_height = self.create_beam()


  def create_beam(self):
        """This method is called when the canvas is reset and cleared, such as when the window resizes, or the canvas is added to a form."""
        canvas = self.canvas_1
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
        
        # Return beam position and dimensions
        return x_start, y_start, beam_length, beam_height

  def draw_boundary_conditions(self, x_start, y_start, beam_length, beam_height):
        conditions = {
            "Free": self.draw_free,
            "Simply supported": self.draw_simply_supported,
            "Fixed": self.draw_fixed
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
        canvas = self.canvas_1
        canvas.fill_style = "#000000"
        # Draw a triangle for simply supported condition
        canvas.begin_path()
        canvas.move_to(x, y + beam_height)
        canvas.line_to(x - 15, y + beam_height + 30)
        canvas.line_to(x + 15, y + beam_height + 30)
        canvas.close_path()
        canvas.fill()

  def draw_fixed(self, x, y, beam_height):
    canvas = self.canvas_1
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

  def Input_click(self, **event_args):
    """This method is called when the button is clicked"""
    self.E=self.input_E.text
    self.I=self.input_I.text
    self.L=self.input_L.text
    self.P=self.input_I.text
    self.I=self.input_I.text
    self.I=self.input_I.text



  


