import sys
import os
import numpy as np
try:
    import PySide2
    #print("PySide2 Detected")
except:
    pass
try:
    import PyQt5
    #print("PySide2 Detected")
except:
    pass

if 'PySide2' in sys.modules:
    from PySide2.QtGui import QPixmap,QPalette,QColor,QPainter, QPen, QBrush, QImage
    from PySide2.QtWidgets import QWidget,QComboBox,QDialog, QCheckBox, QMessageBox, QFileDialog, QPushButton, QLineEdit, QProgressBar, QGridLayout, QHBoxLayout, QVBoxLayout, QApplication, QSplashScreen,QTabWidget,QMainWindow,QLabel,QAction,QStyleFactory,QDialogButtonBox
    from PySide2.QtCore import QTimer,Qt,QSize
elif 'PyQt5' in sys.modules:
    from PyQt5.QtGui import QPixmap,QPalette,QPainter, QPen, QColor, QBrush, QImage
    from PyQt5.QtWidgets import QWidget,QComboBox,QDialog, QCheckBox, QMessageBox, QFileDialog, QPushButton, QLineEdit, QProgressBar, QGridLayout, QHBoxLayout, QVBoxLayout, QApplication, QSplashScreen,QTabWidget,QMainWindow,QLabel,QAction,QStyleFactory,QDialogButtonBox
    from PyQt5.QtCore import QTimer,Qt,QSize,QColor
else:
    sys.exit()
    #print("Missing PySide2 or PyQt5")



class Main(QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        #Initialize the LOGO 
        self.FILE_NAME = 'logo.png'
        self.FOLDER_NAME = '/app_data'
        self.flashSplash(self.FILE_NAME,self.FOLDER_NAME)

        #Setup basics
        self.setWindowTitle("Semantic Segmentation V0.1")
        self.setFixedSize(QSize(1024, 768))
        self.styleSheetList = QStyleFactory.keys()
        self.colorPalettes()
        self.createMenus()


    def createMenus(self):
        ThemeActions = []
        for i,style in enumerate(self.styleSheetList):          
            theme = QAction(str(style), self)
            ThemeActions.append(theme)
            #theme.setShortcut("Ctrl+Q")
            theme.setStatusTip('Set '+str(style)+' Theme')
            s = str(style)
            theme.triggered.connect((lambda  s=s : lambda :self.setUserStyle(s))())
        
        dark_plt = QAction(str('Dark Palette'), self)
        #dark_plt.setShortcut("Ctrl+Q")
        dark_plt.setStatusTip('Apply Dark Color Palette to existing Theme')
        dark_plt.triggered.connect(self.setDarkPalette)

        light_plt = QAction(str('Light Palette'), self)
        #light_plt.setShortcut("Ctrl+Q")
        light_plt.setStatusTip('Apply Dark Color Palette to existing Theme')
        light_plt.triggered.connect(self.setLightPalette)


        One_2DImage = QAction(str('One 2D Image'), self)
        #One_2DImage.setShortcut("Ctrl+Q")
        One_2DImage.setStatusTip('Open one 2D Image')
        #One_2DImage.triggered.connect(self.setOne_2DImage)

        Multiple_2DImages = QAction(str('Multiple 2D Images'), self)
        #Multiple_2DImages.setShortcut("Ctrl+Q")
        Multiple_2DImages.setStatusTip('Select Multiple 2D Images')
        #Multiple_2DImages.triggered.connect(self.setMultiple_2DImages)

        Folder_2DImage = QAction(str('Select a Folder'), self)
        #Folder_2DImage.setShortcut("Ctrl+Q")
        Folder_2DImage.setStatusTip('Select folder containing 2D Images')
        #Folder_2DImage.triggered.connect(self.setFolder_2DImage)

        One_3DImage = QAction(str('One 3D Image'), self)
        #One_3DImage.setShortcut("Ctrl+Q")
        One_3DImage.setStatusTip('Open one 3D Image')
        #One_3DImage.triggered.connect(self.setOne_3DImage)

        Multiple_3DImages = QAction(str('Multiple 3D Images'), self)
        #Multiple_3DImages.setShortcut("Ctrl+Q")
        Multiple_3DImages.setStatusTip('Select Multiple 3D Images')
        #Multiple_3DImages.triggered.connect(self.setMultiple_3DImages)

        Folder_3DImage = QAction(str('Select a Folder'), self)
        #Folder_3DImage.setShortcut("Ctrl+Q")
        Folder_3DImage.setStatusTip('Select folder containing 3D Images')
        #Folder_3DImage.triggered.connect(self.setFolder_3DImage)

        One_Video = QAction(str('One Video'), self)
        #One_Video.setShortcut("Ctrl+Q")
        One_Video.setStatusTip('Open one Video')
        #One_Video.triggered.connect(self.setOne_Video)

        Multiple_Videos = QAction(str('Multiple Videos'), self)
        #Multiple_Videos.setShortcut("Ctrl+Q")
        Multiple_Videos.setStatusTip('Select Multiple Videos')
        #Multiple_Videos.triggered.connect(self.setMultiple_Videos)

        Folder_Video = QAction(str('Select a Folder'), self)
        #Folder_Video.setShortcut("Ctrl+Q")
        Folder_Video.setStatusTip('Select folder containing Videos')
        #Folder_Video.triggered.connect(self.setFolder_Video)

        Output_folder = QAction(str('Set Output Folder'), self)
        #Output_folder.setShortcut("Ctrl+Q")
        Output_folder.setStatusTip('Set Output Folder Location')
        #Output_folder.triggered.connect(self.setOutput_folder)

        Hyperparameter = QAction(str('Set Hyperparameters'), self)
        #Hyperparameter.setShortcut("Ctrl+Q")
        Hyperparameter.setStatusTip('Set Hyperparameters for Machine Learning')
        #Hyperparameter.triggered.connect(self.setHyperparameter)


        Training = QAction(str('Model Training'), self)
        #Training.setShortcut("Ctrl+Q")
        Training.setStatusTip('Set Mode for Traning Machine Learning Model')
        Training.setCheckable(True) 
        #Training.triggered.connect(self.setMode)

        Testing = QAction(str('Model Testing'), self)
        #Testing.setShortcut("Ctrl+Q")
        Testing.setStatusTip('Set Mode for Testing Machine Learning Model')
        Testing.setCheckable(True)
        #Testing.triggered.connect(self.setMode)

        SettingsMenu = self.menuBar().addMenu(str("Settings"))
        ThemeMenu = SettingsMenu.addMenu(str("Theme"))
        for theme in ThemeActions:
            ThemeMenu.addAction(theme)

        ColorPalette = SettingsMenu.addMenu(str("Window Color Palette"))
        ColorPalette.addAction(dark_plt)
        ColorPalette.addAction(light_plt)

        ModeMenu = self.menuBar().addMenu(str("Modes"))
        ModeMenu.addAction(Training)
        ModeMenu.addAction(Testing)


        InputMenu = self.menuBar().addMenu(str("Input"))
        ImageInputMenu = InputMenu.addMenu(str("Image Input"))
        Image2DInputMenu = ImageInputMenu.addMenu(str("2D Image"))
        Image2DInputMenu.addAction(One_2DImage)
        Image2DInputMenu.addAction(Multiple_2DImages)
        Image2DInputMenu.addAction(Folder_2DImage)
        
        Image3DInputMenu = ImageInputMenu.addMenu(str("3D Image"))
        Image3DInputMenu.addAction(One_3DImage)
        Image3DInputMenu.addAction(Multiple_3DImages)
        Image3DInputMenu.addAction(Folder_3DImage)
        
        VideoInputMenu = InputMenu.addMenu(str("Video Input"))
        VideoInputMenu.addAction(One_Video)
        VideoInputMenu.addAction(Multiple_Videos)
        VideoInputMenu.addAction(Folder_Video)
        
        OutputMenu = self.menuBar().addMenu(str("Output"))
        OutputMenu.addAction(Output_folder)

        HyperparameterMenu = self.menuBar().addMenu(str("Hyperparameters"))
        HyperparameterMenu.addAction(Hyperparameter)





        '''
        On_Images = QAction(str('On Image'), self)
        #proj_foldr.setShortcut("Ctrl+Q")
        On_Images.setStatusTip('Computer Vision Process on Images')
        On_Images.triggered.connect(self.setProjectFolder)

        proj_page = QAction(str('Project Page'), self)
        #proj_page.setShortcut("Ctrl+Q")
        proj_page.setStatusTip('Open Project Page')
        proj_page.triggered.connect(self.showProjectPage)

        wifi_page = QAction(str('Wi-Fi Page (only for KIT)'), self)
        #wifi_page.setShortcut("Ctrl+Q")
        wifi_page.setStatusTip('Open Wi-Fi Page')
        wifi_page.triggered.connect(self.showWifiPage)

        gps_page = QAction(str('GPS Page (only for KIT)'), self)
        #gps_page.setShortcut("Ctrl+Q")
        gps_page.setStatusTip('Open GPS Page')
        gps_page.triggered.connect(self.showGPSPage)

        daq_page = QAction(str('DAQ Page (only for KIT)'), self)
        #daq_page.setShortcut("Ctrl+Q")
        daq_page.setStatusTip('Open GPS Page')
        daq_page.triggered.connect(self.showDAQPage)

        roller_page = QAction(str('Roller'), self)
        #roller_page.setShortcut("Ctrl+Q")
        roller_page.setStatusTip('Set Roller Parameters')
        roller_page.triggered.connect(self.showRollerPage)

        material_page = QAction(str('Material'), self)
        #material_page.setShortcut("Ctrl+Q")
        material_page.setStatusTip('Set Material Parameters')
        material_page.triggered.connect(self.showMaterialPage)

        environment_page = QAction(str('Environment'), self)
        #environment_page.setShortcut("Ctrl+Q")
        environment_page.setStatusTip('Set Environment Parameters')
        environment_page.triggered.connect(self.showEnvironmentPage)

        run_page = QAction(str('Run'), self)
        #run_page.setShortcut("Ctrl+Q")
        run_page.setStatusTip('Open Run Page')
        run_page.triggered.connect(self.showRunPage)

        analyze_page = QAction(str('Analyze Data'), self)
        #analyze_page.setShortcut("Ctrl+Q")
        analyze_page.setStatusTip('Open Analyze Page')
        analyze_page.triggered.connect(self.showAnalyzePage)

        shutdown_kit = QAction(str('Shutdown'), self)
        #shutdown_kit.setShortcut("Ctrl+Q")
        shutdown_kit.setStatusTip('Shutdown !?')
        shutdown_kit.triggered.connect(self.shutdown)

        InputMenu = self.menuBar().addMenu(str("Project"))
        InputMenu.addAction(proj_foldr)
        InputMenu.addAction(proj_page)
        InputMenu.addAction(analyze_page)
        '''


        '''

        ParameterMenu = self.menuBar().addMenu(str("Set Parameters"))
        ParameterMenu.addAction(roller_page)
        ParameterMenu.addAction(material_page)
        ParameterMenu.addAction(environment_page)

        ControlMenu = self.menuBar().addMenu(str("Control"))
        ControlMenu.addAction(shutdown_kit)

        self.menuBar().addAction(run_page)
        '''

    def flashSplash(self, FILE_NAME, FOLDER_NAME):
        FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__))+FOLDER_NAME,FILE_NAME)
        
        self.splash = QSplashScreen(QPixmap(FILE_PATH))

        self.splash.show()

        # Close SplashScreen after 5 seconds (5000 ms)
        QTimer.singleShot(5000, self.splash.close)
        # Splash Screen will close after opening Main widget
        self.splash.finish(self)

    def setUserStyle(self, style):     
        app.setStyle(style)
        self.statusBar().showMessage("Selected Style = {}".format(style))


    def setDarkPalette(self):
        app.setPalette(self.dark_palette)
        self.statusBar().showMessage("Selected Palette = DARK")

    def setLightPalette(self):
        app.setPalette(self.default_palette)
        self.statusBar().showMessage("Selected Palette = LIGHT")

    def colorPalettes(self):
        self.default_palette = QPalette()
        self.dark_palette = QPalette()
        self.dark_palette = QPalette()
        self.dark_palette.setColor(QPalette.Background, QColor(50, 50, 50))
        self.dark_palette.setColor(QPalette.Window, QColor(50, 50, 50))
        #self.dark_palette.setColor(QPalette.WindowText, Qt.white)
        self.dark_palette.setColor(QPalette.WindowText, QColor(243, 227, 0))
        self.dark_palette.setColor(QPalette.Base, QColor(0, 0, 0))
        self.dark_palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
        self.dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        self.dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        self.dark_palette.setColor(QPalette.Text, Qt.white)
        self.dark_palette.setColor(QPalette.Button, QColor(50, 50, 50))
        #self.dark_palette.setColor(QPalette.ButtonText, Qt.white)
        self.dark_palette.setColor(QPalette.ButtonText, QColor(243, 227, 0))
        self.dark_palette.setColor(QPalette.BrightText, Qt.red)
        self.dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        self.dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        self.dark_palette.setColor(QPalette.HighlightedText, Qt.black)



    def closedown(self):
        try:
            pass # Exit Function
        except:
            print("exit function warning !!")
        print ("you just closed Semantic Segmentation!!! you are awesome!!!")



if __name__ == "__main__":
    sys.argv.append('--no-sandbox')
    app = QApplication(sys.argv)
    QApplication.processEvents()
    widget = Main()
    app.setApplicationName("SPARC IC Kit")
    widget.show()

    app.exec_()
    sys.exit(widget.closedown())