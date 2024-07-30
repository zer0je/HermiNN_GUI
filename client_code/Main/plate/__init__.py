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

    # Attach change event handler for boundary condition dropdown
    self.boundary_condition.set_event_handler('change', self.boundary_condition_change)

    # Initialize plate
    self.platefigure_reset()
    self.canvas_progress_reset(0)

  def boundary_condition_change(self, **event_args):
        """This method is called when the selected value of the dropdown changes"""
        self.platefigure_reset()

  def platefigure_reset(self, **event_args):
    """This method is called when the canvas is reset and cleared, such as when the window resizes, or the canvas is added to a form."""
    x_start, y_start, plate_width, plate_height=self.create_plate()
    self.draw_boundary_conditions(x_start, y_start, plate_width, plate_height)

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
    canvas.fill_style = "#D3D3D3"  
    canvas.fill_rect(x_start, y_start, plate_width, plate_height)
    # Draw x-axis
    x_axis_length = plate_width / 2
    x_axis_start_x = x_start + plate_width / 2
    x_axis_start_y = y_start + plate_height / 2
    x_axis_end_x = x_axis_start_x + x_axis_length

    canvas.stroke_style = "#000000"  
    canvas.begin_path()
    canvas.move_to(x_axis_start_x, x_axis_start_y)
    canvas.line_to(x_axis_end_x, x_axis_start_y)
    canvas.stroke()

    # Draw x-axis arrow
    arrow_size = 10
    canvas.begin_path()
    canvas.move_to(x_axis_end_x, x_axis_start_y)
    canvas.line_to(x_axis_end_x - arrow_size, x_axis_start_y - arrow_size / 2)
    canvas.move_to(x_axis_end_x, x_axis_start_y)
    canvas.line_to(x_axis_end_x - arrow_size, x_axis_start_y + arrow_size / 2)
    canvas.stroke()

    # Draw x-axis label
    canvas.fill_style = "#000000"  # Black color for text
    canvas.font = "14px Arial"
    canvas.fill_text("x", x_axis_end_x -10, x_axis_start_y +15)

    # Draw y-axis
    y_axis_length = plate_height / 2
    y_axis_start_x = x_start + plate_width / 2
    y_axis_start_y = y_start + plate_height / 2
    y_axis_end_y = y_axis_start_y - y_axis_length

    canvas.begin_path()
    canvas.move_to(y_axis_start_x, y_axis_start_y)
    canvas.line_to(y_axis_start_x, y_axis_end_y)
    canvas.stroke()

    # Draw y-axis arrow
    canvas.begin_path()
    canvas.move_to(y_axis_start_x, y_axis_end_y)
    canvas.line_to(y_axis_start_x - arrow_size / 2, y_axis_end_y + arrow_size)
    canvas.move_to(y_axis_start_x, y_axis_end_y)
    canvas.line_to(y_axis_start_x + arrow_size / 2, y_axis_end_y + arrow_size)
    canvas.stroke()

    # Draw y-axis label
    canvas.fill_text("y", y_axis_start_x-15, y_axis_end_y +15)

    # Return plate position and dimensions
    return x_start, y_start, plate_width, plate_height

  def draw_boundary_conditions(self, x_start, y_start, plate_width, plate_height):
        condition = self.boundary_condition.selected_value
        if condition == "Clamped":
            self.draw_clamped(x_start, y_start, plate_width, plate_height)
        elif condition == "Simply supported":
            self.draw_simply_supported(x_start, y_start, plate_width, plate_height)

  def draw_clamped(self, x, y, plate_width, plate_height):
        canvas = self.platefigure
        canvas.stroke_style = "#000000"
        canvas.line_width = 3
        
        # Draw diagonal lines along the edges to indicate clamped condition
        line_spacing = 10  # spacing between the lines
        line_length = 10  # length of each line
        corner_gap = 0  # gap at the corners
    
        # Top edge
        for i in range(0, plate_width, line_spacing):
            if i < corner_gap or i > plate_width - corner_gap - line_spacing:
                continue
            canvas.begin_path()
            canvas.move_to(x + i, y)
            canvas.line_to(x + i + line_length, y - line_length)
            canvas.stroke()
    
        # Bottom edge
        for i in range(0, plate_width, line_spacing):
            if i < corner_gap or i > plate_width - corner_gap - line_spacing:
                continue
            canvas.begin_path()
            canvas.move_to(x + i, y + plate_height)
            canvas.line_to(x + i + line_length, y + plate_height + line_length)
            canvas.stroke()
    
        # Left edge
        for i in range(0, plate_height, line_spacing):
            if i < corner_gap or i > plate_height - corner_gap - line_spacing:
                continue
            canvas.begin_path()
            canvas.move_to(x, y + i)
            canvas.line_to(x - line_length, y + i + line_length)
            canvas.stroke()
    
        # Right edge
        for i in range(0, plate_height, line_spacing):
            if i < corner_gap or i > plate_height - corner_gap - line_spacing:
                continue
            canvas.begin_path()
            canvas.move_to(x + plate_width, y + i)
            canvas.line_to(x + plate_width + line_length, y + i + line_length)
            canvas.stroke()

  def draw_simply_supported(self, x, y, plate_width, plate_height):
        canvas = self.platefigure
        canvas.fill_style = "#000000"
        
        triangle_size = 8  # Adjust the size of the triangles
        spacing = 20  # Adjust the spacing between triangles
    
        # Top edge
        for i in range(5, plate_width, spacing):
            canvas.begin_path()
            canvas.move_to(x + i, y)
            canvas.line_to(x + i - triangle_size, y - triangle_size)
            canvas.line_to(x + i + triangle_size, y - triangle_size)
            canvas.close_path()
            canvas.fill()
    
        # Bottom edge
        for i in range(1, plate_width, spacing):
            canvas.begin_path()
            canvas.move_to(x + i, y + plate_height)
            canvas.line_to(x + i - triangle_size, y + plate_height + triangle_size)
            canvas.line_to(x + i + triangle_size, y + plate_height + triangle_size)
            canvas.close_path()
            canvas.fill()
    
        # Left edge
        for i in range(1, plate_height, spacing):
            canvas.begin_path()
            canvas.move_to(x, y + i)
            canvas.line_to(x - triangle_size, y + i - triangle_size)
            canvas.line_to(x - triangle_size, y + i + triangle_size)
            canvas.close_path()
            canvas.fill()
    
        # Right edge
        for i in range(1, plate_height, spacing):
            canvas.begin_path()
            canvas.move_to(x + plate_width, y + i)
            canvas.line_to(x + plate_width + triangle_size, y + i - triangle_size)
            canvas.line_to(x + plate_width + triangle_size, y + i + triangle_size)
            canvas.close_path()
            canvas.fill()

  
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
    self.image_plate_deflection.source = image_3d
    self.image_plate_deflection.width = "1000px"
    self.image_plate_deflection.height = "800px"
    
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






  



