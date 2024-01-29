# The PEP 484 type hints stub file for the QtWebSockets module.
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
from PyQt6 import QtNetwork

# Support for QDate, QDateTime and QTime.
import datetime

# Convenient type aliases.
PYQT_SIGNAL = typing.Union[QtCore.pyqtSignal, QtCore.pyqtBoundSignal]
PYQT_SLOT = typing.Union[typing.Callable[..., Any], QtCore.pyqtBoundSignal]


class QMaskGenerator(QtCore.QObject):

    def __init__(self, parent: typing.Optional[QtCore.QObject] = ...) -> None: ...

    def nextMask(self) -> int: ...
    def seed(self) -> bool: ...


class QWebSocket(QtCore.QObject):

    def __init__(self, origin: typing.Optional[str] = ..., version: 'QWebSocketProtocol.Version' = ..., parent: typing.Optional[QtCore.QObject] = ...) -> None: ...

    authenticationRequired: typing.ClassVar[QtCore.pyqtSignal]
    errorOccurred: typing.ClassVar[QtCore.pyqtSignal]
    def subprotocol(self) -> str: ...
    def handshakeOptions(self) -> 'QWebSocketHandshakeOptions': ...
    handshakeInterruptedOnError: typing.ClassVar[QtCore.pyqtSignal]
    alertReceived: typing.ClassVar[QtCore.pyqtSignal]
    alertSent: typing.ClassVar[QtCore.pyqtSignal]
    peerVerifyError: typing.ClassVar[QtCore.pyqtSignal]
    def continueInterruptedHandshake(self) -> None: ...
    @staticmethod
    def maxOutgoingFrameSize() -> int: ...
    def outgoingFrameSize(self) -> int: ...
    def setOutgoingFrameSize(self, outgoingFrameSize: int) -> None: ...
    @staticmethod
    def maxIncomingFrameSize() -> int: ...
    @staticmethod
    def maxIncomingMessageSize() -> int: ...
    def maxAllowedIncomingMessageSize(self) -> int: ...
    def setMaxAllowedIncomingMessageSize(self, maxAllowedIncomingMessageSize: int) -> None: ...
    def maxAllowedIncomingFrameSize(self) -> int: ...
    def setMaxAllowedIncomingFrameSize(self, maxAllowedIncomingFrameSize: int) -> None: ...
    def bytesToWrite(self) -> int: ...
    preSharedKeyAuthenticationRequired: typing.ClassVar[QtCore.pyqtSignal]
    sslErrors: typing.ClassVar[QtCore.pyqtSignal]
    bytesWritten: typing.ClassVar[QtCore.pyqtSignal]
    pong: typing.ClassVar[QtCore.pyqtSignal]
    binaryMessageReceived: typing.ClassVar[QtCore.pyqtSignal]
    textMessageReceived: typing.ClassVar[QtCore.pyqtSignal]
    binaryFrameReceived: typing.ClassVar[QtCore.pyqtSignal]
    textFrameReceived: typing.ClassVar[QtCore.pyqtSignal]
    readChannelFinished: typing.ClassVar[QtCore.pyqtSignal]
    proxyAuthenticationRequired: typing.ClassVar[QtCore.pyqtSignal]
    stateChanged: typing.ClassVar[QtCore.pyqtSignal]
    disconnected: typing.ClassVar[QtCore.pyqtSignal]
    connected: typing.ClassVar[QtCore.pyqtSignal]
    aboutToClose: typing.ClassVar[QtCore.pyqtSignal]
    def ping(self, payload: typing.Union[QtCore.QByteArray, bytes, bytearray, memoryview] = ...) -> None: ...
    @typing.overload
    def open(self, request: QtNetwork.QNetworkRequest, options: 'QWebSocketHandshakeOptions') -> None: ...
    @typing.overload
    def open(self, url: QtCore.QUrl, options: 'QWebSocketHandshakeOptions') -> None: ...
    @typing.overload
    def open(self, url: QtCore.QUrl) -> None: ...
    @typing.overload
    def open(self, request: QtNetwork.QNetworkRequest) -> None: ...
    def close(self, closeCode: 'QWebSocketProtocol.CloseCode' = ..., reason: typing.Optional[str] = ...) -> None: ...
    def request(self) -> QtNetwork.QNetworkRequest: ...
    def sslConfiguration(self) -> QtNetwork.QSslConfiguration: ...
    def setSslConfiguration(self, sslConfiguration: QtNetwork.QSslConfiguration) -> None: ...
    @typing.overload
    def ignoreSslErrors(self, errors: typing.Iterable[QtNetwork.QSslError]) -> None: ...
    @typing.overload
    def ignoreSslErrors(self) -> None: ...
    def sendBinaryMessage(self, data: typing.Union[QtCore.QByteArray, bytes, bytearray, memoryview]) -> int: ...
    def sendTextMessage(self, message: typing.Optional[str]) -> int: ...
    def closeReason(self) -> str: ...
    def closeCode(self) -> 'QWebSocketProtocol.CloseCode': ...
    def origin(self) -> str: ...
    def requestUrl(self) -> QtCore.QUrl: ...
    def resourceName(self) -> str: ...
    def version(self) -> 'QWebSocketProtocol.Version': ...
    def state(self) -> QtNetwork.QAbstractSocket.SocketState: ...
    def setPauseMode(self, pauseMode: QtNetwork.QAbstractSocket.PauseMode) -> None: ...
    def resume(self) -> None: ...
    def setReadBufferSize(self, size: int) -> None: ...
    def readBufferSize(self) -> int: ...
    def maskGenerator(self) -> typing.Optional[QMaskGenerator]: ...
    def setMaskGenerator(self, maskGenerator: typing.Optional[QMaskGenerator]) -> None: ...
    def setProxy(self, networkProxy: QtNetwork.QNetworkProxy) -> None: ...
    def proxy(self) -> QtNetwork.QNetworkProxy: ...
    def peerPort(self) -> int: ...
    def peerName(self) -> str: ...
    def peerAddress(self) -> QtNetwork.QHostAddress: ...
    def pauseMode(self) -> QtNetwork.QAbstractSocket.PauseMode: ...
    def localPort(self) -> int: ...
    def localAddress(self) -> QtNetwork.QHostAddress: ...
    def isValid(self) -> bool: ...
    def flush(self) -> bool: ...
    def errorString(self) -> str: ...
    error: typing.ClassVar[QtCore.pyqtSignal]
    def abort(self) -> None: ...


class QWebSocketCorsAuthenticator(PyQt6.sip.simplewrapper):

    @typing.overload
    def __init__(self, origin: typing.Optional[str]) -> None: ...
    @typing.overload
    def __init__(self, other: 'QWebSocketCorsAuthenticator') -> None: ...

    def allowed(self) -> bool: ...
    def setAllowed(self, allowed: bool) -> None: ...
    def origin(self) -> str: ...
    def swap(self, other: 'QWebSocketCorsAuthenticator') -> None: ...


class QWebSocketHandshakeOptions(PyQt6.sip.simplewrapper):

    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, other: 'QWebSocketHandshakeOptions') -> None: ...

    def __eq__(self, other: object): ...
    def __ne__(self, other: object): ...
    def setSubprotocols(self, protocols: typing.Iterable[typing.Optional[str]]) -> None: ...
    def subprotocols(self) -> typing.List[str]: ...
    def swap(self, other: 'QWebSocketHandshakeOptions') -> None: ...


class QWebSocketProtocol(PyQt6.sip.simplewrapper):

    class CloseCode(enum.Enum):
        CloseCodeNormal = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeGoingAway = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeProtocolError = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeDatatypeNotSupported = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeReserved1004 = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeMissingStatusCode = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeAbnormalDisconnection = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeWrongDatatype = ... # type: QWebSocketProtocol.CloseCode
        CloseCodePolicyViolated = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeTooMuchData = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeMissingExtension = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeBadOperation = ... # type: QWebSocketProtocol.CloseCode
        CloseCodeTlsHandshakeFailed = ... # type: QWebSocketProtocol.CloseCode

    class Version(enum.Enum):
        VersionUnknown = ... # type: QWebSocketProtocol.Version
        Version0 = ... # type: QWebSocketProtocol.Version
        Version4 = ... # type: QWebSocketProtocol.Version
        Version5 = ... # type: QWebSocketProtocol.Version
        Version6 = ... # type: QWebSocketProtocol.Version
        Version7 = ... # type: QWebSocketProtocol.Version
        Version8 = ... # type: QWebSocketProtocol.Version
        Version13 = ... # type: QWebSocketProtocol.Version
        VersionLatest = ... # type: QWebSocketProtocol.Version


class QWebSocketServer(QtCore.QObject):

    class SslMode(enum.Enum):
        SecureMode = ... # type: QWebSocketServer.SslMode
        NonSecureMode = ... # type: QWebSocketServer.SslMode

    def __init__(self, serverName: typing.Optional[str], secureMode: 'QWebSocketServer.SslMode', parent: typing.Optional[QtCore.QObject] = ...) -> None: ...

    def supportedSubprotocols(self) -> typing.List[str]: ...
    def setSupportedSubprotocols(self, protocols: typing.Iterable[typing.Optional[str]]) -> None: ...
    def handshakeTimeoutMS(self) -> int: ...
    def setHandshakeTimeout(self, msec: int) -> None: ...
    preSharedKeyAuthenticationRequired: typing.ClassVar[QtCore.pyqtSignal]
    closed: typing.ClassVar[QtCore.pyqtSignal]
    sslErrors: typing.ClassVar[QtCore.pyqtSignal]
    peerVerifyError: typing.ClassVar[QtCore.pyqtSignal]
    newConnection: typing.ClassVar[QtCore.pyqtSignal]
    originAuthenticationRequired: typing.ClassVar[QtCore.pyqtSignal]
    serverError: typing.ClassVar[QtCore.pyqtSignal]
    acceptError: typing.ClassVar[QtCore.pyqtSignal]
    def handleConnection(self, socket: typing.Optional[QtNetwork.QTcpSocket]) -> None: ...
    def serverUrl(self) -> QtCore.QUrl: ...
    def supportedVersions(self) -> typing.List[QWebSocketProtocol.Version]: ...
    def sslConfiguration(self) -> QtNetwork.QSslConfiguration: ...
    def setSslConfiguration(self, sslConfiguration: QtNetwork.QSslConfiguration) -> None: ...
    def proxy(self) -> QtNetwork.QNetworkProxy: ...
    def setProxy(self, networkProxy: QtNetwork.QNetworkProxy) -> None: ...
    def serverName(self) -> str: ...
    def setServerName(self, serverName: typing.Optional[str]) -> None: ...
    def resumeAccepting(self) -> None: ...
    def pauseAccepting(self) -> None: ...
    def errorString(self) -> str: ...
    def error(self) -> QWebSocketProtocol.CloseCode: ...
    def nextPendingConnection(self) -> typing.Optional[QWebSocket]: ...
    def hasPendingConnections(self) -> bool: ...
    def socketDescriptor(self) -> PyQt6.sip.voidptr: ...
    def setSocketDescriptor(self, socketDescriptor: PyQt6.sip.voidptr) -> bool: ...
    def secureMode(self) -> 'QWebSocketServer.SslMode': ...
    def serverAddress(self) -> QtNetwork.QHostAddress: ...
    def serverPort(self) -> int: ...
    def maxPendingConnections(self) -> int: ...
    def setMaxPendingConnections(self, numConnections: int) -> None: ...
    def isListening(self) -> bool: ...
    def close(self) -> None: ...
    def listen(self, address: typing.Union[QtNetwork.QHostAddress, QtNetwork.QHostAddress.SpecialAddress] = ..., port: int = ...) -> bool: ...