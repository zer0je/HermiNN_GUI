'''
GUI Built by Yeong Je Kim
email: satiho@snu.ac.kr

Architecture Built by Jinkyo Han
email: 38jinkyo@snu.ac.kr
'''

from ._anvil_designer import MainTemplate
from anvil import *
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.server
from .beam import beam
from .plate import plate


class Main(MainTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run before the form opens.

  def button_1_click(self, **event_args):
    """This method is called when the button is clicked"""
    self.content_panel.clear()  
    self.content_panel.add_component(beam())  

  def button_2_click(self, **event_args):
    """This method is called when the button is clicked"""
    self.content_panel.clear()  
    self.content_panel.add_component(plate())  

