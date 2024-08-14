from ._anvil_designer import beamTemplate
from anvil import *
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import plotly.graph_objects as go
import anvil.server
import time


class beam(beamTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Set Color
    self.button_calculate.background = "#CED8F6"
    self.button_Px.background="#CED8F6"
    self.button_Px_undo.background="#CED8F6"
    self.button_q.background="#CED8F6"


    # Initialize dropdown menus
    self.left_boundary_condition.items = [ "Pinned", "Fixed","Free"]
    self.right_boundary_condition.items = [ "Pinned", "Fixed","Free"]

    # Attach change event handlers
    self.left_boundary_condition.set_event_handler('change',self.left_boundary_condition_change)
    self.right_boundary_condition.set_event_handler('change',self.right_boundary_condition_change)

    # 나타나기 효과
    self.text_result.visible=False
    
    # Iniate canvas drawing
    self.beamfigure_reset()
    self.canvas_progress_reset(0)

    # Iniate List
    self.P=[]
    self.x_p=[]
    self.q_l=[]
    self.x_l=[]
    self.q_r=[]
    self.x_r=[]

    
  def beamfigure_reset(self, **event_args):
        """This method is called when the canvas is reset and cleared, such as when the window resizes, or the canvas is added to a form."""
        x_start, y_start, beam_length, beam_height = self.create_beam()
        self.draw_boundary_conditions(x_start, y_start, beam_length, beam_height)


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
        canvas.fill_style = "#000089"  
        canvas.fill_rect(x_start, y_start, beam_length, beam_height)

        # Draw arrow indicating direction (rightward arrow at the end of the beam)
        arrow_length = 80
        arrow_height = 20
        x_arrow_start = x_start + beam_length
        y_arrow_start = y_start + beam_height / 2

        canvas.stroke_style = "#000000"
        canvas.begin_path()
        canvas.move_to(x_arrow_start, y_arrow_start)
        canvas.line_to(x_arrow_start + arrow_length, y_arrow_start)
        canvas.line_to(x_arrow_start + arrow_length - arrow_height / 2, y_arrow_start - arrow_height / 2)
        canvas.move_to(x_arrow_start + arrow_length, y_arrow_start)
        canvas.line_to(x_arrow_start + arrow_length - arrow_height / 2, y_arrow_start + arrow_height / 2)
        canvas.stroke()
    
        # Draw x-axis label
        canvas.fill_style = "#000000"
        canvas.font = "14px Arial"
        canvas.fill_text("x", x_arrow_start + arrow_length + 10, y_arrow_start + 5)

        # Draw vertical z-axis at the left end of the beam
        z_arrow_length = 80
        z_arrow_height = 20
        z_arrow_start_x = x_start
        z_arrow_start_y = y_start + beam_height / 2
    
        canvas.begin_path()
        canvas.move_to(z_arrow_start_x, z_arrow_start_y)
        canvas.line_to(z_arrow_start_x, z_arrow_start_y + z_arrow_length)
        canvas.line_to(z_arrow_start_x - z_arrow_height / 2, z_arrow_start_y + z_arrow_length - z_arrow_height / 2)
        canvas.move_to(z_arrow_start_x, z_arrow_start_y + z_arrow_length)
        canvas.line_to(z_arrow_start_x + z_arrow_height / 2, z_arrow_start_y + z_arrow_length - z_arrow_height / 2)
        canvas.stroke()
        
        # Draw z-axis label
        canvas.fill_text("z", z_arrow_start_x-2 , z_arrow_start_y + z_arrow_length + 20)

        # Return beam position and dimensions
        return x_start, y_start, beam_length, beam_height

  def draw_boundary_conditions(self, x_start, y_start, beam_length, beam_height):
        conditions = {
            "Pinned": self.draw_simply_supported,
            "Fixed": self.draw_fixed,  
            "Free": self.draw_free,
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

  def convert_boundary_condition_value(self,condition):
    if condition == "Free":
        return 'x'
    elif condition == "Pinned":
        return 's'
    elif condition == "Fixed":
        return 'f'
      
  def button_Px_click(self,**event_args):
    P=self.input_P.text if self.input_P.text else '0'
    x_p=self.input_x_p.text if self.input_x_p.text else '0'
    self.P.append(P)
    self.x_p.append(x_p)
    
    # Beam을 다시 그리기
    x_start, y_start, beam_length, beam_height = self.create_beam()
    self.draw_boundary_conditions(x_start, y_start, beam_length, beam_height)
    
    # 하중 표시
    self.draw_point_load(x_start, y_start, beam_length, beam_height)

  def button_Px_undo_click(self, **event_args):
    if self.P and self.x_p:
      self.P.pop()
      self.x_p.pop()
      # Beam을 다시 그리기
      x_start, y_start, beam_length, beam_height = self.create_beam()
      self.draw_boundary_conditions(x_start, y_start, beam_length, beam_height)
        
      # 남아 있는 모든 하중을 다시 그리기
      self.draw_point_load(x_start, y_start, beam_length, beam_height)


  def draw_point_load(self, x_start, y_start, beam_length, beam_height):
    canvas = self.beamfigure

    L=float(self.input_L.text)
    P_array=list(map(float,self.P))
    x_p_array=list(map(float,self.x_p))

    for P, x_p in zip(P_array,x_p_array):
      # 하중 위치 계산 (x_p는 0에서 L 사이의 값이라고 가정)
      load_position = x_start + x_p / L * beam_length
  
      # 하중 크기와 위치 시각화
      load_arrow_length = 50  # 하중 화살표 길이
      load_arrow_height = 10  # 하중 화살표 높이
      
      # 하중 화살표 그리기
      canvas.stroke_style = "#FF0000"  # 빨간색으로 표시
      canvas.begin_path()
      canvas.move_to(load_position, y_start)
      canvas.line_to(load_position, y_start + load_arrow_length)
      canvas.line_to(load_position - load_arrow_height / 2, y_start + load_arrow_length - load_arrow_height / 2)
      canvas.move_to(load_position, y_start + load_arrow_length)
      canvas.line_to(load_position + load_arrow_height / 2, y_start + load_arrow_length - load_arrow_height / 2)
      canvas.stroke()
  
      # 하중 크기 텍스트 표시
      canvas.fill_style = "#000000"  # 검은색으로 표시
      canvas.font = "12px Arial"
      canvas.fill_text(f"{P}N", load_position + 5, y_start + load_arrow_length + 10)

  def button_q_click(self, **event_args):
    q_l=self.input_q_l.text if self.input_q_l.text else '0'
    x_l=self.input_x_l.text if self.input_x_l.text else '0'
    q_r=self.input_q_r.text if self.input_q_r.text else '0'
    x_r=self.input_x_r.text if self.input_x_r.text else self.input_L.text
    self.q_l.append(q_l)
    self.x_l.append(x_l)
    self.q_r.append(q_r)
    self.x_r.append(x_r)
  
  def button_calculate_click(self, **event_args):
    """This method is called when the button is clicked"""
    left_condition=self.convert_boundary_condition_value(self.left_boundary_condition.selected_value)
    right_condition=self.convert_boundary_condition_value(self.right_boundary_condition.selected_value)
    self.E=self.input_E.text if self.input_E.text else '206e09'
    self.I=self.input_I.text if self.input_I.text else '10000'
    self.L=self.input_L.text if self.input_L.text else '1'
    self.lr=self.input_lr.text if self.input_lr.text else '0.1'
    self.epochs=self.input_epochs.text if self.input_epochs.text else '10'

     # 백그라운드 태스크 시작
    task_id = anvil.server.call(
            'launch_calculate_beam', left_condition, right_condition, self.E, self.I, self.L, self.P, self.x_p, self.q_l,self.x_l,self.q_r,self.x_r, self.lr, int(self.epochs)
        )

    # 하중 초기화
    self.P.clear()
    self.x_p.clear()
    self.q_l.clear()
    self.x_l.clear()
    self.q_r.clear()
    self.x_r.clear()
    
    # Beam을 다시 그리기
    x_start, y_start, beam_length, beam_height = self.create_beam()
    self.draw_boundary_conditions(x_start, y_start, beam_length, beam_height)

     # 진행 상황 폴링
    while True:
        progress_data = anvil.server.call('get_task_progress', task_id)
        progress = progress_data.get('progress', 0)
        self.canvas_progress_reset(progress)
      
        if not progress_data['running']:
          result_text = progress_data['result_text']
          break
          
    image_media=anvil.server.call('create_image',"/tmp/beam_plot.png")
    self.image_beam_deflection.source = image_media
    self.image_beam_deflection.width = "1000px"  
    self.image_beam_deflection.height = "400px"

    self.text_result.visible=True
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

  



  


    






  


