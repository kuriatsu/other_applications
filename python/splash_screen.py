#!/usr/bin/env python3
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, Pango

class Splash(Gtk.Window):

	def __init__(self):
		Gtk.Window.__init__(self, title="splashtitle")
		maingrid = Gtk.Grid()
		self.add(maingrid)
		maingrid.set_border_width(80)
		# set text for the spash window
		label = Gtk.Label("Eat bananas while you are waiting for Firefox")
		# label.modify_font(Pango.FontDescription('Ubuntu 22'))
		maingrid.attach(label, 0, 0, 1, 1)

def splashwindow():
    window = Splash()
    window.set_decorated(False)
    window.set_resizable(False)
    window.set_position(Gtk.WindowPosition.CENTER)
    window.show_all()
    Gtk.main()

splashwindow()
