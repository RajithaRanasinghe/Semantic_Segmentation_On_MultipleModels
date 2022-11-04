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
    from PySide2.QtGui import QPixmap,QPalette,QColor,QPainter, QPen, QBrush, QImage, QIcon, QIntValidator, QDoubleValidator
    from PySide2.QtWidgets import QWidget,QComboBox,QDialog, QCheckBox, QMessageBox, QFileDialog, QPushButton, QLineEdit, QProgressBar, QGridLayout, QHBoxLayout, QVBoxLayout, QApplication, QSplashScreen,QTabWidget,QMainWindow,QLabel,QAction,QStyleFactory,QDialogButtonBox
    from PySide2.QtCore import QTimer,Qt,QSize
elif 'PyQt5' in sys.modules:
    from PyQt5.QtGui import QPixmap,QPalette,QPainter, QPen, QColor, QBrush, QImage, QIcon, QIntValidator, QDoubleValidator
    from PyQt5.QtWidgets import QWidget,QComboBox,QDialog, QCheckBox, QMessageBox, QFileDialog, QPushButton, QLineEdit, QProgressBar, QGridLayout, QHBoxLayout, QVBoxLayout, QApplication, QSplashScreen,QTabWidget,QMainWindow,QLabel,QAction,QStyleFactory,QDialogButtonBox
    from PyQt5.QtCore import QTimer,Qt,QSize,QColor
else:
    sys.exit()
    #print("Missing PySide2 or PyQt5")


from skimage.filters import threshold_yen, threshold_triangle, threshold_otsu, threshold_minimum, threshold_mean, threshold_isodata
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
import Unet

class Main(QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        #Initialize the banners
        self.ICON_NAME = 'icon.png'
        self.LOGO_NAME = 'logo.png'
        self.FOLDER_NAME = '/app_data'
        self.flashSplash(self.LOGO_NAME,self.FOLDER_NAME)

        #Setup basics
        self.setWindowTitle("Semantic Segmentation V0.1")
        ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__))+self.FOLDER_NAME, self.ICON_NAME)
        self.setWindowIcon(QIcon(ICON_PATH))
        self.setFixedSize(QSize(1024, 768))
        self.styleSheetList = QStyleFactory.keys()

        #parameters
        self.image_dir = ''
        self.mask_dir = ''
        self.val_image_dir = ''
        self.val_mask_dir = ''

        self.parameters = {'epoch':200, 'datasetName':'dataset', 'LearningRate':5e-4, 'BatchSize':4, 'ModelInputHeight':256, 'ModelInputWidth':256, 'LogPath':'', 'TrainingImages':'', 'TrainingMasks':'', 'ValidationImages':'', 'ValidationMasks':'', 'OutputPath':'', 'Mode':'Training', 'Model':0, 'ModelName':''}
        self.statusDict = {'Mode':'None', 'Input':'None', 'Pre_Process':'None', 'ML_Model':'None', 'Post_Process':'None', 'Output':'None'}
        self.thresholdingMethodList = {'Yen’s method' : threshold_yen, 'triangle algorithm' : threshold_triangle , 'Otsu’s method' : threshold_otsu, 'minimum method' : threshold_minimum, 'mean of grayscale values' : threshold_mean,'ISODATA method' : threshold_isodata}
        self.Ml_ModelList = {'U-net':Unet, 'FCN-Resnet50':fcn_resnet50, 'FCN-Resnet101':fcn_resnet101, 'DeepLabV3-Resnet50':deeplabv3_resnet50, 'DeepLabV3-Resnet101':deeplabv3_resnet101, 'DeepLabV3-MobileNet-V3-Large':deeplabv3_mobilenet_v3_large, 'Lraspp-MobileNet-V3-Large':lraspp_mobilenet_v3_large}

        self.PreProcessLayers = ['None']
        self.PostProcessLayers = ['None']
        self.ModelLayers = ['None']
        for key in self.thresholdingMethodList.keys():
            self.PreProcessLayers.append(key)
            self.PostProcessLayers.append(key)

        for key in self.Ml_ModelList.keys():
            self.ModelLayers.append(key)
        
        self.MLModels = []
        
        self.colorPalettes()
        self.createMenus()
        self.initUI()
    
        print(self.PreProcessLayers)


    def createUIcomponents(self):
        self.Mode_label = QLabel("Mode = {}".format(self.statusDict['Mode']))
        self.Input_label = QLabel("Input = {}".format(self.statusDict['Input']))
        self.PreProcess_label = QLabel("Pre Process = {}".format(self.statusDict['Pre_Process']))
        self.MLModel_label = QLabel("ML Model = {}".format(self.statusDict['ML_Model']))
        self.PostProcess_label = QLabel("Post Process = {}".format(self.statusDict['Post_Process']))
        self.Output_label = QLabel("Output = {}".format(self.statusDict['Output']))

        # Hyperparameter Inputs
        self.Epoch_label = QLabel("No Epochs : ")
        self.Epoch_label.setWordWrap(True)
        self.Epoch_Text = QLineEdit('200')
        self.Epoch_Text.setAlignment(Qt.AlignRight)
        self.Epoch_Text.setValidator(QIntValidator(0,1000))
        self.Epoch_Text.textChanged.connect(self.updateStatus)

        self.Batch_label = QLabel("Batch Size : ")
        self.Batch_label.setWordWrap(True)
        self.Batch_Text = QLineEdit('4')
        self.Batch_Text.setAlignment(Qt.AlignRight)
        self.Batch_Text.setValidator(QIntValidator(0,1000))
        self.Batch_Text.textChanged.connect(self.updateStatus)

        self.ModelInHeight_label = QLabel("Model Input Height : ")
        self.ModelInHeight_label.setWordWrap(True)
        self.ModelInHeight_Text = QLineEdit('256')
        self.ModelInHeight_Text.setAlignment(Qt.AlignRight)
        self.ModelInHeight_Text.setValidator(QIntValidator(0,1000))
        self.ModelInHeight_Text.textChanged.connect(self.updateStatus)

        self.ModelInWidth_label = QLabel("Model Input Width : ")
        self.ModelInWidth_label.setWordWrap(True)
        self.ModelInWidth_Text = QLineEdit('256')
        self.ModelInWidth_Text.setAlignment(Qt.AlignRight)
        self.ModelInWidth_Text.setValidator(QIntValidator(0,1000))
        self.ModelInWidth_Text.textChanged.connect(self.updateStatus)

        self.LearningRate_label = QLabel("Learning Rate : ")
        self.LearningRate_label.setWordWrap(True)
        self.LearningRate_Text = QLineEdit('5e-4')
        self.LearningRate_Text.setAlignment(Qt.AlignRight)
        self.LearningRate_Text.setValidator(QDoubleValidator(0,10,1000000))
        self.LearningRate_Text.textChanged.connect(self.updateStatus)

        self.DataName_label = QLabel("Dataset Name : ")
        self.DataName_label.setWordWrap(True)
        self.DataName_Text = QLineEdit('dataset')
        self.DataName_Text.setAlignment(Qt.AlignRight)
        self.DataName_Text.textChanged.connect(self.updateStatus)



    def initUI(self):
        #Layouts        
        self.main_layout = QVBoxLayout()
        self.status_layout = QHBoxLayout()
        self.hyperparameters_layout = QHBoxLayout()
        
        self.data_layout = QGridLayout()
        self.imgView_layout = QVBoxLayout()
        self.control_btn_layout = QHBoxLayout()


        #set margins (left,top,right,bottom)
        self.main_layout.setContentsMargins(1,1,1,1)
        self.status_layout.setContentsMargins(1,1,1,1)
        self.hyperparameters_layout.setContentsMargins(1,1,1,1)
        
        
        self.data_layout.setContentsMargins(1,1,1,1)
        self.control_btn_layout.setContentsMargins(1,1,1,1)
        self.imgView_layout.setContentsMargins(1,1,1,1)

        #layout Widgets
        self.status_widget = QWidget()
        self.hyperparameters_widget = QWidget()


        self.data_widget = QWidget()
        self.control_btn_widget = QWidget()
        self.main_widget = QWidget()
        self.imgView_widget = QWidget()

        #create components for layout
        self.createUIcomponents()

        #Add component widgets to status layout
        self.status_layout.addWidget(self.Mode_label)
        self.status_layout.addWidget(self.Input_label)
        self.status_layout.addWidget(self.PreProcess_label)
        self.status_layout.addWidget(self.MLModel_label)
        self.status_layout.addWidget(self.PostProcess_label)
        self.status_layout.addWidget(self.Output_label)

        #Set widget
        self.status_widget.setLayout(self.status_layout)

        #Add component widgets to hyperparameter layout
        self.hyperparameters_layout.addWidget(self.Epoch_label)
        self.hyperparameters_layout.addWidget(self.Epoch_Text)
        self.hyperparameters_layout.addWidget(self.Batch_label)
        self.hyperparameters_layout.addWidget(self.Batch_Text)
        self.hyperparameters_layout.addWidget(self.ModelInHeight_label)
        self.hyperparameters_layout.addWidget(self.ModelInHeight_Text)
        self.hyperparameters_layout.addWidget(self.ModelInWidth_label)
        self.hyperparameters_layout.addWidget(self.ModelInWidth_Text)
        self.hyperparameters_layout.addWidget(self.LearningRate_label)
        self.hyperparameters_layout.addWidget(self.LearningRate_Text)
        self.hyperparameters_layout.addWidget(self.DataName_label)
        self.hyperparameters_layout.addWidget(self.DataName_Text)

        #Set widget
        self.hyperparameters_widget.setLayout(self.hyperparameters_layout)


        self.imgView_widget.setLayout(self.imgView_layout)


        #Add widjets to main layout
        self.main_layout.addWidget(self.status_widget)
        self.main_layout.addWidget(self.hyperparameters_widget)

        self.main_layout.addWidget(self.data_widget)
        self.main_layout.addWidget(self.imgView_widget)
        self.main_layout.addWidget(self.control_btn_widget)

        #Set widget
        self.main_widget.setLayout(self.main_layout)

        self.setCentralWidget(self.main_widget)       


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
        One_2DImage.triggered.connect(self.setOne_2DImage)

        Multiple_2DImages = QAction(str('Multiple 2D Images'), self)
        #Multiple_2DImages.setShortcut("Ctrl+Q")
        Multiple_2DImages.setStatusTip('Select Multiple 2D Images')
        Multiple_2DImages.triggered.connect(self.setMultiple_2DImages)

        Folder_2DImage = QAction(str('Select a Folder'), self)
        #Folder_2DImage.setShortcut("Ctrl+Q")
        Folder_2DImage.setStatusTip('Select folder containing 2D Images')
        Folder_2DImage.triggered.connect(self.setFolder_2DImage)

        One_3DImage = QAction(str('One 3D Image'), self)
        #One_3DImage.setShortcut("Ctrl+Q")
        One_3DImage.setStatusTip('Open one 3D Image')
        One_3DImage.triggered.connect(self.setOne_3DImage)

        Multiple_3DImages = QAction(str('Multiple 3D Images'), self)
        #Multiple_3DImages.setShortcut("Ctrl+Q")
        Multiple_3DImages.setStatusTip('Select Multiple 3D Images')
        Multiple_3DImages.triggered.connect(self.setMultiple_3DImages)

        Folder_3DImage = QAction(str('Select a Folder'), self)
        #Folder_3DImage.setShortcut("Ctrl+Q")
        Folder_3DImage.setStatusTip('Select folder containing 3D Images')
        Folder_3DImage.triggered.connect(self.setFolder_3DImage)

        One_Video = QAction(str('One Video'), self)
        #One_Video.setShortcut("Ctrl+Q")
        One_Video.setStatusTip('Open one Video')
        One_Video.triggered.connect(self.setOne_Video)

        Multiple_Videos = QAction(str('Multiple Videos'), self)
        #Multiple_Videos.setShortcut("Ctrl+Q")
        Multiple_Videos.setStatusTip('Select Multiple Videos')
        Multiple_Videos.triggered.connect(self.setMultiple_Videos)

        Folder_Video = QAction(str('Select a Folder'), self)
        #Folder_Video.setShortcut("Ctrl+Q")
        Folder_Video.setStatusTip('Select folder containing Videos')
        Folder_Video.triggered.connect(self.setFolder_Video)

        Output_folder = QAction(str('Set Output Folder'), self)
        #Output_folder.setShortcut("Ctrl+Q")
        Output_folder.setStatusTip('Set Output Folder Location')
        Output_folder.triggered.connect(self.setOutput_folder)

        TrImages = QAction(str('Training Images'), self)
        #TrImages.setShortcut("Ctrl+Q")
        TrImages.setStatusTip('Set Training Image Folder')
        TrImages.triggered.connect(self.setFolder_TrImages)

        TrMasks = QAction(str('Training Masks'), self)
        #TrMasks.setShortcut("Ctrl+Q")
        TrMasks.setStatusTip('Set Training Image Folder')
        TrMasks.triggered.connect(self.setFolder_Video)

        ValImages = QAction(str('Validation Images'), self)
        #ValImages.setShortcut("Ctrl+Q")
        ValImages.setStatusTip('Set Validation Image Folder')
        ValImages.triggered.connect(self.setFolder_Video)


        ValMasks = QAction(str('Validation Masks'), self)
        #ValMasks.setShortcut("Ctrl+Q")
        ValMasks.setStatusTip('Set Validation Masks Folder')
        ValMasks.triggered.connect(self.setFolder_Video)




        self.Training = QAction(str('Model Training'), self)
        #Training.setShortcut("Ctrl+Q")
        self.Training.setStatusTip('Set Mode for Traning Machine Learning Model')
        self.Training.setCheckable(True) 
        self.Training.triggered.connect(lambda checked : self.setMode('Training') if checked else self.setMode('None'))

        self.Testing = QAction(str('Model Testing'), self)
        #Testing.setShortcut("Ctrl+Q")
        self.Testing.setStatusTip('Set Mode for Testing Machine Learning Model')
        self.Testing.setCheckable(True)
        self.Testing.triggered.connect(lambda checked : self.setMode('Testing') if checked else self.setMode('None'))

        SettingsMenu = self.menuBar().addMenu(str("Settings"))
        ThemeMenu = SettingsMenu.addMenu(str("Theme"))
        for theme in ThemeActions:
            ThemeMenu.addAction(theme)

        ColorPalette = SettingsMenu.addMenu(str("Window Color Palette"))
        ColorPalette.addAction(dark_plt)
        ColorPalette.addAction(light_plt)

        ModeMenu = self.menuBar().addMenu(str("Modes"))
        ModeMenu.addAction(self.Training)
        ModeMenu.addAction(self.Testing)

        TrainInputMenu = self.menuBar().addMenu(str("Train Input"))
        TrainImages = TrainInputMenu.addMenu(str("Train"))
        TrainImages.addAction(TrImages)
        TrainImages.addAction(TrMasks)
        ValidationImages = TrainInputMenu.addMenu(str("Validation"))
        ValidationImages.addAction(ValImages)
        ValidationImages.addAction(ValMasks)


        InputMenu = self.menuBar().addMenu(str("Test Input"))
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

        self.PreProcessActions = []
        for i,item in enumerate(self.PreProcessLayers):          
            PreProcess = QAction(str(item), self)
            self.PreProcessActions.append(PreProcess)
            #theme.setShortcut("Ctrl+Q")
            PreProcess.setStatusTip(str(PreProcess))
            PreProcess.setCheckable(True)
            s = str(item)
            PreProcess.triggered.connect((lambda  s=s : lambda :self.setPreProcessLayer(s))())

        PreProcess = self.menuBar().addMenu(str("Pre Processing"))
        for item in self.PreProcessActions:
            PreProcess.addAction(item)

        self.ModelActions = []
        for i,item in enumerate(self.ModelLayers):          
            model = QAction(str(item), self)
            self.ModelActions.append(model)
            #theme.setShortcut("Ctrl+Q")
            model.setStatusTip(str(model))
            model.setCheckable(True)
            s = str(item)
            model.triggered.connect((lambda  s=s : lambda :self.setModelLayer(s))())

        Models = self.menuBar().addMenu(str("ML Models"))
        for item in self.ModelActions:
            Models.addAction(item)

        self.PostProcessActions = []
        for i,item in enumerate(self.PostProcessLayers):          
            PostProcess = QAction(str(item), self)
            self.PostProcessActions.append(PostProcess)
            #theme.setShortcut("Ctrl+Q")
            PostProcess.setStatusTip(str(model))
            PostProcess.setCheckable(True)
            s = str(item)
            PostProcess.triggered.connect((lambda  s=s : lambda :self.setPostProcessLayer(s))())

        PostProcess = self.menuBar().addMenu(str("Post Processing"))
        for item in self.PostProcessActions:
            PostProcess.addAction(item)
        
        OutputMenu = self.menuBar().addMenu(str("Output"))
        OutputMenu.addAction(Output_folder)


    def setPostProcessLayer(self, PostProcess):
        for item in  self.PostProcessActions:
            if item.text() == PostProcess:
                item.setChecked(True)
            else:
                item.setChecked(False)
            self.statusDict['Post_Process'] = PostProcess
        self.updateStatus()

    def setModelLayer(self, Model):
        for item in self.ModelActions:
            if item.text() == Model:
                item.setChecked(True)
            else:
                item.setChecked(False)
            self.statusDict['ML_Model'] = Model
        self.updateStatus()

    def setPreProcessLayer(self, PreProcess):
        for item in self.PreProcessActions:
            if item.text() == PreProcess:
                item.setChecked(True)
            else:
                item.setChecked(False)
        self.statusDict['Pre_Process'] = PreProcess
        self.updateStatus()



    def setFolder_2DImage(self):
        FOLDER_NAME, FOLDER_PATH = self.getFolder()
        self.statusDict['Input'] = '2D Image Folder'
        self.updateStatus()
        print(FOLDER_NAME)
        print(FOLDER_PATH)


    def setMultiple_2DImages(self):
        FILE_NAMES, FILE_PATHS = self.getMultipleFiles()
        self.statusDict['Input'] = 'Multiple 2D Images'
        self.updateStatus()
        print(FILE_NAMES)
        print(FILE_PATHS)

    def setOne_2DImage(self):
        FILE_NAME, FILE_PATH = self.getFile()
        self.statusDict['Input'] = 'One 2D Image'
        self.updateStatus()
        print(FILE_NAME)
        print(FILE_PATH)

    def setFolder_3DImage(self):
        FOLDER_NAME, FOLDER_PATH = self.getFolder()
        self.statusDict['Input'] = '3D Image Folder'
        self.updateStatus()
        print(FOLDER_NAME)
        print(FOLDER_PATH)


    def setMultiple_3DImages(self):
        FILE_NAMES, FILE_PATHS = self.getMultipleFiles()
        self.statusDict['Input'] = 'Multiple 3D Images'
        self.updateStatus()
        print(FILE_NAMES)
        print(FILE_PATHS)

    def setOne_3DImage(self):
        FILE_NAME, FILE_PATH = self.getFile()
        self.statusDict['Input'] = 'One 3D Image'
        self.updateStatus()
        print(FILE_NAME)
        print(FILE_PATH)

    def setOne_Video(self):
        FILE_NAME, FILE_PATH = self.getFile()
        self.statusDict['Input'] = 'One Video'
        self.updateStatus()
        print(FILE_NAME)
        print(FILE_PATH)

    def setMultiple_Videos(self):
        FILE_NAMES, FILE_PATHS = self.getMultipleFiles()
        self.statusDict['Input'] = 'Multiple Videos'
        self.updateStatus()
        print(FILE_NAMES)
        print(FILE_PATHS)

    def setFolder_Video(self):
        FOLDER_NAME, FOLDER_PATH = self.getFolder()
        self.statusDict['Input'] = 'Video Folder'
        self.updateStatus()
        print(FOLDER_NAME)
        print(FOLDER_PATH)

    def setOutput_folder(self):
        FOLDER_NAME, FOLDER_PATH = self.getFolder()
        self.statusDict['Output'] = 'Output Folder : {}'.format(FOLDER_NAME)
        self.updateStatus()
        print(FOLDER_NAME)
        print(FOLDER_PATH)


    def setFolder_TrImages(self):
        FOLDER_NAME, FOLDER_PATH = self.getFolder()
        self.statusDict['Input'] = 'Training Images'

        self.updateStatus()


    def setMode(self, Mode):
        self.statusDict['Mode'] = Mode
        if self.statusDict['Mode'] == 'Testing':
                self.Testing.setChecked(True)
                self.Training.setChecked(False)
        elif self.statusDict['Mode'] == 'Training':
                self.Testing.setChecked(False)
                self.Training.setChecked(True)
        else:
                self.Testing.setChecked(False)
                self.Training.setChecked(False)
        self.updateStatus()


    def updateStatus(self):
        self.Mode_label.setText("Mode = {}".format(self.statusDict['Mode']))
        self.Input_label.setText("Input = {}".format(self.statusDict['Input']))
        self.PreProcess_label.setText("Pre Process = {}".format(self.statusDict['Pre_Process']))
        self.MLModel_label.setText("ML Model = {}".format(self.statusDict['ML_Model']))
        self.PostProcess_label.setText("Post Process = {}".format(self.statusDict['Post_Process']))
        self.Output_label.setText("Output = {}".format(self.statusDict['Output']))

        #self.parameters = {'epoch':200, 'datasetName':'dataset', 'LearningRate':5e-4, 'BatchSize':4, 'ModelInputHeight':256, 'ModelInputWidth':256, 'LogPath':'', 'TrainingImages':'', 'TrainingMasks':'', 'ValidationImages':'', 'ValidationMasks':'', 'OutputPath':'', 'Mode':'Training', 'Model':0, 'ModelName':''}
        self.parameters['datasetName'] = str(self.DataName_Text.text())
        self.parameters['epoch'] = int(self.Epoch_Text.text())
        self.parameters['LearningRate'] = float(self.LearningRate_Text.text())
        self.parameters['BatchSize'] = int(self.Batch_Text.text())
        self.parameters['ModelInputHeight'] = int(self.ModelInHeight_Text.text())
        self.parameters['ModelInputWidth'] = int(self.ModelInWidth_Text.text())
        self.parameters['Mode'] = self.statusDict['Mode']
        self.parameters['ModelName'] = self.statusDict['ML_Model']
        self.parameters['Model'] =  self.Ml_ModelList[self.parameters['ModelName']]
        self.parameters['TrainingImages'] = self.image_dir
        self.parameters['TrainingMasks'] = self.mask_dir
        self.parameters['ValidationImages'] = self.val_image_dir
        self.parameters['ValidationMasks'] = self.val_mask_dir
    
    
    def getFolder(self):
        dlgBox = QFileDialog()
        dlgBox.setFileMode(QFileDialog.Directory)
        FOLDER_PATH = dlgBox.getExistingDirectory(self,'Open Folder')
        FOLDER_NAME = FOLDER_PATH.split('/')[-1]
        return FOLDER_NAME, FOLDER_PATH

    def getFile(self):
        dlgBox = QFileDialog()
        dlgBox.setFileMode(QFileDialog.ExistingFile)
        FILE_PATH,_ = dlgBox.getOpenFileName(self, 'Open File')
        FILE_NAME = FILE_PATH.split('/')[-1]
        return FILE_NAME, FILE_PATH 

    def getMultipleFiles(self):
        dlgBox = QFileDialog()
        dlgBox.setFileMode(QFileDialog.ExistingFiles)
        FILE_PATHS,_ = dlgBox.getOpenFileNames(self, 'Open Files')
        FILE_NAMES = []
        for path in FILE_PATHS:
            FILE_NAMES.append(path.split('/')[-1])
        return FILE_NAMES, FILE_PATHS        


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