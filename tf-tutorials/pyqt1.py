from PyQt4 import QtCore,uic,QtGui 
import sys
from PyQt4.Qt import QStringList, QListWidget

if __name__=="__main__":
    print "main"
    app=QtGui.QApplication(sys.argv)
    str=QStringList()
    
    str << "ok" << "nice"
    
    list=QtGui.QListWidget()
    list.addItems(str)
    list.show()
    
    
    combo=QtGui.QComboBox()
    combo.addItems(str)
    combo.show()
    
    print sys.exit(app.exec_())
    