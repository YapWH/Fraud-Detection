from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QFileDialog, QDesktopWidget, QVBoxLayout, QWidget, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt, QPropertyAnimation, QPoint

import bg_rc
import main


class Ui_FraudDetection(object):
    def setupUi(self, FraudDetection):
        FraudDetection.setObjectName("FraudDetection")
        FraudDetection.resize(1087, 672)
        FraudDetection.setStyleSheet("background-color: rgba(255, 244, 203, 70")
        self.centralwidget = QtWidgets.QWidget(FraudDetection)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(370, 40, 361, 101))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(42)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(0,0,0);")
        self.label.setObjectName("label")
        font1 = QtGui.QFont("Arial", 15)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(130, 180, 761, 61))
        self.textEdit.setStyleSheet("border:2px solid rgba(0,0,0,0);\n"
"border-bottom-color:rgba(0,0,0,100);\n"
"background-color:rgba(0,0,0,0);")
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setFont(font1)
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(130, 280, 761, 61))
        self.textEdit_2.setStyleSheet("border:2px solid rgba(0,0,0,0);\n"
"border-bottom-color:rgba(0,0,0,100);\n"
"background-color:rgba(0,0,0,0);")
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_2.setFont(font1)
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(190, 380, 180, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        self.checkBox.setFont(font)
        self.checkBox.setStyleSheet("background-color: rgb(255, 244, 203);\n"
"color:rgb(255,255,255);")
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(190, 440, 160, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        self.checkBox_2.setFont(font)
        self.checkBox_2.setStyleSheet("background-color: rgb(255, 244, 203);\n"
"color:rgb(255,255,255);")
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_2.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(720, 510, 61, 61))
        self.pushButton.setStyleSheet("QPushButton{\n"
"    \n"
"    image: url(:/tencent.png);\n"
"    background-color:rgba(0,0,0,0);\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    paddin-bottom:3px;\n"
"}")
        self.pushButton.setText("")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.openQQ)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(800, 510, 61, 61))
        self.pushButton_2.setStyleSheet("QPushButton{\n"
"    \n"
"    \n"
"    image: url(:/wechat.png);\n"
"    background-color:rgba(0,0,0,0);\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    paddin-bottom:3px;\n"
"}")
        self.pushButton_2.setText("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.openWechat)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(890, 510, 61, 61))
        self.pushButton_3.setStyleSheet("QPushButton{\n"
"    \n"
"    \n"
"    image: url(:/facebook.png);\n"
"    background-color:rgba(0,0,0,0);\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    paddin-bottom:3px;\n"
"}")
        self.pushButton_3.setText("")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.openFacebook)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(970, 510, 61, 61))
        self.pushButton_4.setStyleSheet("QPushButton{\n"
"    image: url(:/google.png);\n"
"    background-color:rgba(0,0,0,0);\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    paddin-bottom:3px;\n"
"}")
        self.pushButton_4.setText("")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.openGoogle)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(700, 440, 111, 61))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color:rgba(0,0,0,0);")
        self.label_2.setObjectName("label_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(490, 380, 71, 71))
        self.pushButton_5.setStyleSheet("QPushButton{\n"
"    \n"
"    image: url(:/right-arrow.png);\n"
"    background-color:rgba(0,0,0,0);\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    paddin-bottom:3px;\n"
"}")
        self.pushButton_5.setText("")
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.login)
        FraudDetection.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(FraudDetection)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1087, 26))
        self.menubar.setObjectName("menubar")
        self.menuMain = QtWidgets.QMenu(self.menubar)
        self.menuMain.setObjectName("menuMain")
        self.menuLog_in = QtWidgets.QMenu(self.menubar)
        self.menuLog_in.setObjectName("menuLog_in")
        self.menuDonate = QtWidgets.QMenu(self.menubar)
        self.menuDonate.setObjectName("menuDonate")
        FraudDetection.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(FraudDetection)
        self.statusbar.setObjectName("statusbar")
        FraudDetection.setStatusBar(self.statusbar)
        self.actionFraud_Detection = QtWidgets.QAction(FraudDetection)
        self.actionFraud_Detection.setObjectName("actionFraud_Detection")
        self.actionFraud_Detection.triggered.connect(self.open_upload_data_window)
        self.actionContact_Us = QtWidgets.QAction(FraudDetection)
        self.actionContact_Us.setObjectName("actionContact_Us")
        self.actionAbout_Us = QtWidgets.QAction(FraudDetection)
        self.actionAbout_Us.setObjectName("actionAbout_Us")
        self.actionVIP_Account = QtWidgets.QAction(FraudDetection)
        self.actionVIP_Account.setObjectName("actionVIP_Account")
        self.actionLog_in_as_Guest = QtWidgets.QAction(FraudDetection)
        self.actionLog_in_as_Guest.setObjectName("actionLog_in_as_Guest")
        self.actionAbout_Donation = QtWidgets.QAction(FraudDetection)
        self.actionAbout_Donation.setObjectName("actionAbout_Donation")
        self.menuMain.addSeparator()
        self.menuMain.addAction(self.actionFraud_Detection)
        self.menuMain.addSeparator()
        self.menuMain.addAction(self.actionContact_Us)
        self.menuMain.addSeparator()
        self.menuMain.addAction(self.actionAbout_Us)
        self.menuMain.addSeparator()
        self.menuLog_in.addAction(self.actionVIP_Account)
        self.menuLog_in.addSeparator()
        self.menuLog_in.addAction(self.actionLog_in_as_Guest)
        self.menuLog_in.addSeparator()
        self.menuDonate.addSeparator()
        self.menuDonate.addAction(self.actionAbout_Donation)
        self.menuDonate.addSeparator()
        self.menubar.addAction(self.menuLog_in.menuAction())
        self.menubar.addAction(self.menuMain.menuAction())
        self.menubar.addAction(self.menuDonate.menuAction())

        self.retranslateUi(FraudDetection)
        QtCore.QMetaObject.connectSlotsByName(FraudDetection)

    def retranslateUi(self, FraudDetection):
        _translate = QtCore.QCoreApplication.translate
        FraudDetection.setWindowTitle(_translate("FraudDetection", "FraudDetection"))
        self.label.setText(_translate("FraudDetection", "Welcome"))
        self.textEdit.setPlaceholderText(_translate("FraudDetection", "Account"))
        self.textEdit_2.setPlaceholderText(_translate("FraudDetection", "Password"))
        self.checkBox.setText(_translate("FraudDetection", "Remember me"))
        self.checkBox_2.setText(_translate("FraudDetection", "Auto Login"))
        self.label_2.setText(_translate("FraudDetection", "Other Login:"))
        self.menuMain.setTitle(_translate("FraudDetection", "Function"))
        self.menuLog_in.setTitle(_translate("FraudDetection", "Log in"))
        self.menuDonate.setTitle(_translate("FraudDetection", "Donate Us"))
        self.actionFraud_Detection.setText(_translate("FraudDetection", "Fraud Detection"))
        self.actionContact_Us.setText(_translate("FraudDetection", "Contact Us"))
        self.actionAbout_Us.setText(_translate("FraudDetection", "About"))
        self.actionVIP_Account.setText(_translate("FraudDetection", "VIP Account"))
        self.actionLog_in_as_Guest.setText(_translate("FraudDetection", "Log in as Guest"))
        self.actionAbout_Donation.setText(_translate("FraudDetection", "About Donation"))


    def openQQ(self):
        QDesktopServices.openUrl(QtCore.QUrl("https://im.qq.com/"))

    def openWechat(self):
        QDesktopServices.openUrl(QtCore.QUrl("https://www.wechat.com/"))

    def openFacebook(self):
        QDesktopServices.openUrl(QtCore.QUrl("https://www.facebook.com/"))

    def openGoogle(self):
        QDesktopServices.openUrl(QtCore.QUrl("https://www.google.com/"))

    def login(self):
        text = self.textEdit.toPlainText()
        if text.strip() == "CAORUI":
            QMessageBox.information(None, "VIP", "Welcome, esteemed VIP user")
        else:
            QMessageBox.information(None, "Guest", "Log in as Guest user")

    def open_upload_data_window(self):
        self.upload_data_window = frauddetection()
        self.upload_data_window.show()


class frauddetection(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fraud Detection")
        self.setGeometry(415, 180, 1087, 672)
        
        self.setStyleSheet("background-color: #333333;")

        layout = QVBoxLayout()

        self.label_welcome = QLabel("Welcome to use our Fraud Detector!", self)
        self.label_welcome.setFont(QFont("Arial Black", 22))
        self.label_welcome.setStyleSheet("color: white;")
        self.label_welcome.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.label_welcome, alignment=Qt.AlignCenter)

        self.animation_button = QPropertyAnimation(self)
        self.animation_button.setTargetObject(self.label_welcome)
        self.animation_button.setPropertyName(b"pos")
        self.animation_button.setStartValue(QPoint(self.width() / 6, -self.label_welcome.height()))
        self.animation_button.setEndValue(QPoint(self.width() / 6, 100))
        self.animation_button.setDuration(1000)
        self.animation_button.start()

        self.button = QPushButton("Upload Data", self)
        self.button.setFont(QFont("Arial", 14))
        self.button.setStyleSheet("background-color: #169CF5; color: white; border-radius: 10px;")
        button_size = self.button.sizeHint()
        self.button.setFixedSize(button_size.width() * 2, button_size.height() * 2)
        layout.addWidget(self.button, alignment=Qt.AlignCenter)
        
        self.button_1 = QPushButton("execute", self)
        self.button_1.setFont(QFont("Arial", 14))
        self.button_1.setStyleSheet("background-color: #1fef0c; color: white; border-radius: 10px;")
        self.button_1.setFixedSize(button_size.width() * 1.5, button_size.height() * 1.5)
        self.button_1.clicked.connect(self.execute)
        layout.addWidget(self.button_1, alignment=Qt.AlignCenter)

        self.button.clicked.connect(self.upload_data)
        self.label = QLabel("", self)
        self.label.setFont(QFont("Arial", 14))
        self.label.setStyleSheet("color: white;")
        layout.addWidget(self.label, alignment=Qt.AlignCenter)


        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def upload_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_name:
            self.label.setText(file_name)

    def execute(self):
        main.demo()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FraudDetection = QtWidgets.QMainWindow()
    ui = Ui_FraudDetection()
    ui.setupUi(FraudDetection)
    FraudDetection.show()
    sys.exit(app.exec_())
