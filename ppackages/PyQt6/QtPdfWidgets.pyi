# The PEP 484 type hints stub file for the QtPdfWidgets module.
#
# Generated by SIP 6.8.0
#
# Copyright (c) 2023 Riverbank Computing Limited <info@riverbankcomputing.com>
# 
# This file is part of PyQt6.
# 
# This file may be used under the terms of the GNU General Public License
# version 3.0 as published by the Free Software Foundation and appearing in
# the file LICENSE included in the packaging of this file.  Please review the
# following information to ensure the GNU General Public License version 3.0
# requirements will be met: http://www.gnu.org/copyleft/gpl.html.
# 
# If you do not wish to use this file under the terms of the GPL version 3.0
# then you may purchase a commercial license.  For more information contact
# info@riverbankcomputing.com.
# 
# This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
# WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.


import enum
import typing

import PyQt6.sip

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtPdf
from PyQt6 import QtWidgets

# Support for QDate, QDateTime and QTime.
import datetime

# Convenient type aliases.
PYQT_SIGNAL = typing.Union[QtCore.pyqtSignal, QtCore.pyqtBoundSignal]
PYQT_SLOT = typing.Union[typing.Callable[..., Any], QtCore.pyqtBoundSignal]


class QPdfPageSelector(QtWidgets.QWidget):

    def __init__(self, parent: typing.Optional[QtWidgets.QWidget]) -> None: ...

    currentPageLabelChanged: typing.ClassVar[QtCore.pyqtSignal]
    currentPageChanged: typing.ClassVar[QtCore.pyqtSignal]
    documentChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setCurrentPage(self, index: int) -> None: ...
    def currentPageLabel(self) -> str: ...
    def currentPage(self) -> int: ...
    def document(self) -> typing.Optional[QtPdf.QPdfDocument]: ...
    def setDocument(self, document: typing.Optional[QtPdf.QPdfDocument]) -> None: ...


class QPdfView(QtWidgets.QAbstractScrollArea):

    class ZoomMode(enum.Enum):
        Custom = ... # type: QPdfView.ZoomMode
        FitToWidth = ... # type: QPdfView.ZoomMode
        FitInView = ... # type: QPdfView.ZoomMode

    class PageMode(enum.Enum):
        SinglePage = ... # type: QPdfView.PageMode
        MultiPage = ... # type: QPdfView.PageMode

    def __init__(self, parent: typing.Optional[QtWidgets.QWidget]) -> None: ...

    def mouseReleaseEvent(self, event: typing.Optional[QtGui.QMouseEvent]) -> None: ...
    def mouseMoveEvent(self, event: typing.Optional[QtGui.QMouseEvent]) -> None: ...
    def mousePressEvent(self, event: typing.Optional[QtGui.QMouseEvent]) -> None: ...
    currentSearchResultIndexChanged: typing.ClassVar[QtCore.pyqtSignal]
    searchModelChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setCurrentSearchResultIndex(self, currentResult: int) -> None: ...
    def currentSearchResultIndex(self) -> int: ...
    def setSearchModel(self, searchModel: typing.Optional[QtPdf.QPdfSearchModel]) -> None: ...
    def searchModel(self) -> typing.Optional[QtPdf.QPdfSearchModel]: ...
    def scrollContentsBy(self, dx: int, dy: int) -> None: ...
    def resizeEvent(self, event: typing.Optional[QtGui.QResizeEvent]) -> None: ...
    def paintEvent(self, event: typing.Optional[QtGui.QPaintEvent]) -> None: ...
    documentMarginsChanged: typing.ClassVar[QtCore.pyqtSignal]
    pageSpacingChanged: typing.ClassVar[QtCore.pyqtSignal]
    zoomFactorChanged: typing.ClassVar[QtCore.pyqtSignal]
    zoomModeChanged: typing.ClassVar[QtCore.pyqtSignal]
    pageModeChanged: typing.ClassVar[QtCore.pyqtSignal]
    documentChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setZoomFactor(self, factor: float) -> None: ...
    def setZoomMode(self, mode: 'QPdfView.ZoomMode') -> None: ...
    def setPageMode(self, mode: 'QPdfView.PageMode') -> None: ...
    def setDocumentMargins(self, margins: QtCore.QMargins) -> None: ...
    def documentMargins(self) -> QtCore.QMargins: ...
    def setPageSpacing(self, spacing: int) -> None: ...
    def pageSpacing(self) -> int: ...
    def zoomFactor(self) -> float: ...
    def zoomMode(self) -> 'QPdfView.ZoomMode': ...
    def pageMode(self) -> 'QPdfView.PageMode': ...
    def pageNavigator(self) -> typing.Optional[QtPdf.QPdfPageNavigator]: ...
    def document(self) -> typing.Optional[QtPdf.QPdfDocument]: ...
    def setDocument(self, document: typing.Optional[QtPdf.QPdfDocument]) -> None: ...
