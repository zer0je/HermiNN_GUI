from ._anvil_designer import beamTemplate
from anvil import *
import anvil.server


class beam(beamTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)
    self.canvas_1_reset()

    # Any code you write here will run before the form opens.

  def canvas_1_reset(self, **event_args):
        """This method is called when the canvas is reset and cleared, such as when the window resizes, or the canvas is added to a form."""
        canvas = self.canvas_1
        width, height = canvas.get_width(), canvas.get_height()

        # Draw a horizontal beam (1D beam)
        beam_length = width * 0.8  # 80% of the canvas width
        beam_height = 10  # Fixed height for the beam

        # Calculate positions
        x_start = (width - beam_length) / 2
        y_start = height / 2 - beam_height / 2

        # Set drawing style
        canvas.fill_style = "#000000"  # Black color
        canvas.fill_rect(x_start, y_start, beam_length, beam_height)
