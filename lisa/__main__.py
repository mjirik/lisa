# This is because we want to draw splash screen before time-consuming imports
import argparse

parser = argparse.ArgumentParser(
    # Turn off help, so we print all options in response to -h
    add_help=False
)
parser.add_argument(
    '-ni', '--no_interactivity', action='store_true',
    help='run in no interactivity mode, seeds must be defined')
# Read alternative config file. First is loaded default config. Then user
# config in lisa_data directory. After that is readed config defined by
# --configfile parameter
knownargs, unknownargs = parser.parse_known_args()

if knownargs.no_interactivity:
    import organ_segmentation
    organ_segmentation.main()
else:

    import PyQt4
    import PyQt4.QtGui
    import sys
    from PyQt4.QtGui import QApplication
    app = QApplication(sys.argv)
    # Create and display the splash screen
    import splash_screen
    splash = splash_screen.splash_screen(app)
    import organ_segmentation
    organ_segmentation.main(app, splash)
