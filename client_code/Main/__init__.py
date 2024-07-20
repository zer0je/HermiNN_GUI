from ._anvil_designer import MainTemplate
from anvil import *
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.server
from .beam import beam


class Main(MainTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run before the form opens.

  def button_1_click(self, **event_args):
    """This method is called when the button is clicked"""
    self.content_panel.clear()  # 기존 폼의 내용을 지웁니다.
    self.content_panel.add_component(beam())  # 새로운 폼을 추가합니다.

  def button_2_click(self, **event_args):
    """This method is called when the button is clicked"""
    pass

