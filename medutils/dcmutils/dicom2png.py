#!/usr/bin/env python
"""
Convert DICOM files to corresponding PNG files.
NOTE:  Only works with DICOM files containing a single
layer!
"""
import sys, os, collections, toml, asyncio, hashlib, time, platform, queue
import numpy as np, png, pydicom, multiprocessing
from pathlib import Path
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QPushButton,
    QWidget,
    QLabel,
    QGridLayout,
    QPlainTextEdit,
    QFileDialog,
)
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QSize, QTimer


def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)
    sys.exit(1)


def getConfigFileName(path=None):
    if path is None:
        path = os.path.dirname(sys.argv[0])
    config_file_name = ".dicom_to_png.conf"
    full_path = os.path.join(path, config_file_name)
    return full_path


def readConfigFile(path=None):
    config_file = getConfigFileName(path)
    if not os.path.isfile(config_file):
        initializeConfigFile(path)
    config = ""
    with open(config_file, "r") as fin:
        config = toml.loads(fin.read())
    return config


def initializeConfigFile(path=None):
    config_file = getConfigFileName(path)
    default_config = {
        "output_path": os.path.join(os.path.expanduser("~"), "png_files_from_dicom")
    }
    saveConfigToFile(default_config)


def saveConfigToFile(config, path=None):
    config_file = getConfigFileName(path)
    with open(config_file, "w") as fout:
        fout.write(toml.dumps(config))


def needs_rescale(hdr):
    return hasattr(hdr, "RescaleSlope") or hasattr(hdr, "RescaleIntercept")


def rescale_image(img, hdr):
    """Apply rescale formula from DICOM header, if that information is available."""
    if not needs_rescale(hdr):
        return (img, hdr)
    if type(hdr) == type([]):
        hdr = hdr[0]
    img = np.array(img)
    img_type = img.dtype
    # Get the scaling info
    rescale_slope = float(getattr(hdr, "RescaleSlope", 1))
    rescale_intercept = float(getattr(hdr, "RescaleIntercept", 0))
    # Re-Scale
    img = img.astype(np.float64) * rescale_slope + rescale_intercept
    img = img.astype(img_type)
    # Update the header
    setattr(hdr, "RescaleSlope", 1.0)
    setattr(hdr, "RescaleIntercept", 0.0)
    return (img, hdr)


def apply_LUT(img, hdr):
    """
    Apply LUT specified in header to the image, if the header specifies one.
    Specification:
    http://dicom.nema.org/medical/dicom/2017a/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.1
    """
    lut_seq = getattr(hdr, "VOILUTSequence", None)
    if lut_seq is None:
        # print("No LUT for image {}".format(generate_unique_filename(hdr)))
        return img, hdr
    # Use the first available LUT:
    lut_desc = getattr(lut_seq[0], "LUTDescriptor", None)
    lut_data = getattr(lut_seq[0], "LUTData", None)
    if lut_desc is None or lut_data is None:
        return img, hdr
    try:
        first_value = int(lut_desc[1])
    except:
        pass
    bit_depth = int(lut_desc[2])
    sign_selector = "u" if type(first_value) == int and first_value >= 0 else ""
    type_selector = 8
    while type_selector < bit_depth and type_selector < 64:
        type_selector *= 2
    orig_type = img.dtype

    img = np.round(img)

    if type(first_value) != int:
        first_value = img.min()

    LUT = {
        int(v): lut_data[j]
        for j, v in [(i, first_value + i) for i in range(len(lut_data))]
    }

    img2 = np.array(img)
    img2 = img2.astype("{}int{}".format(sign_selector, type_selector))
    img2[img < first_value] = first_value
    img2 = np.vectorize(lambda x: LUT[int(x)])(img2)
    img2[img >= (first_value + len(lut_data))] = lut_data[-1]

    del hdr.VOILUTSequence

    return img2.astype(orig_type), hdr


def apply_window(img, hdr):
    """
    Apply intensity window as defined in the DICOM header to the image (if any window
    is defined).
    This is applied after any LUT and rescale/intercept.
    See https://www.dabsoft.ch/dicom/3/C.11.2.1.2/
    This implementation will set the output range (min, max) equal to the
    input range (original min, max).  If scaling is desired, do that after calling
    this function.
    """
    window_center = getattr(hdr, "WindowCenter", None)
    if window_center is None:
        return img, hdr
    y_min = img.min()
    y_max = img.max()
    window_width = getattr(hdr, "WindowWidth", None)
    window_center, window_width = float(window_center), float(window_width)

    img_out = np.zeros_like(img)

    # y = ((x - (c - 0.5)) / (w-1) + 0.5) * (y max - y min )+ y min
    img_out = ((img - (window_center - 0.5)) / (window_width - 1) + 0.5) * (
        y_max - y_min
    ) + y_min
    #  if (x <= c - 0.5 - (w-1)/2), then y = y min
    img_out[img <= (window_center - 0.5 - (window_width - 1) / 2.0)] = y_min
    # else if (x > c - 0.5 + (w-1)/2), then y = y max ,
    img_out[img > (window_center - 0.5 + (window_width - 1) / 2.0)] = y_max

    return img_out, hdr


def read_dicom_raw(file_path):
    dicom = pydicom.read_file(file_path)
    img = dicom.pixel_array
    return img, dicom


def read_dicom(file_path):
    img, hdr = read_dicom_raw(file_path)
    img, hdr = rescale_image(img, hdr)
    img, hdr = apply_LUT(img, hdr)
    img, hdr = apply_window(img, hdr)
    return img, hdr


def path_to_list(file_path):
    if file_path == "" or file_path == os.path.sep:
        return []
    rest, last = os.path.split(file_path)
    # Check to see if we have hit a "root" (or "drive"), bail out if so.
    if rest == file_path:
        return [rest]
    return path_to_list(rest) + [last] if last != "" else path_to_list(rest)


def abbreviate_path(file_path, length=2):
    path_list = path_to_list(file_path)
    abbrev = file_path
    if len(path_list) > length + 1:
        abbrev = os.path.join("...", *path_list[-length:])
    return abbrev


def win_safe_path(path):
    """
    Remove leading 'slash' in Windows paths, which should be relative or begin with
    a drive letter, not a slash.
    """
    if path is None or path == '':
        return None
    # Sometimes in Windows, you end up with a path like '/C:/foo/bar' --- not sure why.
    if platform.system().lower() == "windows":
        if path[0] in ["/", "\\"]:
            return path[1:]
    return path


def generate_unique_filename(dicom_header, extension=".dcm"):
    """
    Generates and returns a unique filename based on the Patient ID and
    internal unique idientifier contained in the DICOM header metadata.
    Unique names are created in the following format:
        ```text {PatientID}_{InstanceHash}{Extension}
        ```
    where `{PatientID}` is the `Patient ID` field from the DICOM header,
    `{InstanceHash}` is the initial 16 characters of the hexadecimal encoding
    of the *sha-1* hash of the `SOP Instance UID` field from the DICOM header,
    and `{Extension}` is the file extension e.g. `.dcm` for DICOM files and
    `.pngs` for PNG format.
    """
    patient_id = dicom_header.PatientID
    sop_instance_uid = dicom_header.SOPInstanceUID
    instance_hash = hashlib.sha1(sop_instance_uid.encode("utf-8")).hexdigest()[:16]
    return "{}_{}{}".format(patient_id, instance_hash, extension)


class ConverterWindow(QMainWindow):
    NUM_THREADS = multiprocessing.cpu_count()

    sig_abort_workers = pyqtSignal()

    def __init__(self):
        self.platform = platform.system().lower()
        QMainWindow.__init__(self)
        self.config = readConfigFile()
        self.setWindowTitle("DICOM to PNG")
        self.outputDir = self.config["output_path"]
        # we want to drop files onto this window:
        self.setAcceptDrops(True)
        # keep it large enough to be an easy target
        self.setMinimumSize(QSize(400, 600))
        self.responseLines = collections.deque(maxlen=20)
        self.queue = queue.Queue()
        self.working = {}
        self.convertedCount = 0
        self.didAbort = False

        centralWidget = QWidget(self)
        gridLayout = QGridLayout()
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)

        self.configButton = QPushButton("Save to...")
        self.configButton.clicked.connect(self.setOutputDirectory)
        self.configButton.setToolTip(
            "Click to select the folder where PNG images should be placed."
        )

        self.addFilesButton = QPushButton("Add Files")
        self.addFilesButton.clicked.connect(lambda: self.addFilesDialog())
        self.addFilesButton.setToolTip("Click to add one or more DICOM files.")

        self.addDirButton = QPushButton("Add Folder")
        self.addDirButton.clicked.connect(lambda: self.addDirectoryDialog())
        self.addDirButton.setToolTip("Click to add a DICOM folder or folder tree.")

        self.stopButton = QPushButton("Stop")
        self.stopButton.setDisabled(True)
        self.stopButton.clicked.connect(self.abortWorkers)
        self.stopButton.setToolTip("Click to stop all conversions already in progress.")

        self.exitButton = QPushButton("Exit")
        self.exitButton.clicked.connect(self.stopAndExit)
        self.exitButton.setToolTip("Click to exit the application.")

        gridLayout.addWidget(self.configButton, 0, 0, 1, 1)
        gridLayout.addWidget(self.stopButton, 0, 2, 1, 1)
        gridLayout.addWidget(self.exitButton, 0, 3, 1, 1)
        gridLayout.addWidget(self.addFilesButton, 1, 0, 1, 2)
        gridLayout.addWidget(self.addDirButton, 1, 2, 1, 2)

        self.indicateThreadsRunning(False)

        # Give some simple instructions:
        self.instructions = QLabel(
            "Drop DICOM files or directories\nhere to convert to PNG.", self
        )
        self.instructions.setAlignment(QtCore.Qt.AlignCenter)
        gridLayout.addWidget(self.instructions, 2, 0, 1, 4)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        gridLayout.addWidget(self.log, 3, 0, 1, 4)

        self.setStatusBar("Output Folder: {}".format(abbreviate_path(self.outputDir)))

    @pyqtSlot()
    def stopAndExit(self):
        self.exitButton.setText("Exiting...")
        self.exitButton.setEnabled(False)
        self.setStatusBar("Exiting....")
        self.abortWorkers()
        app.quit()

    def addResponse(self, msg):
        self.log.appendPlainText(msg)
        print(msg)

    def setResponse(self, msg):
        self.setStatusBar(msg)
        self.addResponse(msg)

    def setStatusBar(self, msg):
        self.statusBar().showMessage(msg)

    def indicateThreadsRunning(self, flag):
        self.stopButton.setEnabled(flag)
        self.exitButton.setEnabled(not flag)
        self.configButton.setEnabled(not flag)
        msg = "Working..." if flag else "Ready."
        self.setStatusBar(msg)

    def addFilesDialog(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select DICOM Files",
            "",
            "DICOM Files (*.dcm *.dicom *);;All Files (*)",
            options=options,
        )
        new_items = {"files": files, "dirs": []}
        self.processNewItems(new_items)

    def showDirectoryDialog(self, caption="Select DICOM Folder."):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        dirname = QFileDialog.getExistingDirectory(
            self, caption, os.getcwd(), options=options
        )
        return win_safe_path(dirname)

    def addDirectoryDialog(self):
        dirname = self.showDirectoryDialog()
        new_items = {"files": [], "dirs": [dirname]}
        self.processNewItems(new_items)

    def setOutputDirectory(self):
        dialog = QFileDialog(self, "Select destination folder for output images.")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setDirectory(self.outputDir)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        # [KLUDGE] alert:  This one can't use the system dialog on Mac OS because for
        # some reason the "New Folder" button will crash the app.  So on Mac, use the
        # non-native dialog instead for now.
        # Re-visit this in the future to try to remove it...
        if self.platform == "darwin":
            dialog.setOption(QFileDialog.DontUseNativeDialog, True)

        if dialog.exec():
            dirname = win_safe_path(dialog.selectedFiles()[0])
            self.config["output_path"] = dirname
            self.outputDir = dirname
            saveConfigToFile(self.config)
            self.setStatusBar(
                "Output Folder: {}".format(abbreviate_path(self.outputDir))
            )
        else:
            print("Set output folder failed.")

    # --------------------------------------------------
    # Originally inspired by:
    # https://stackoverflow.com/a/8580720
    #
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        new_items = {"files": [], "dirs": []}
        for url in event.mimeData().urls():
            path = win_safe_path(url.path())
            if os.path.isdir(path):
                self.addResponse(
                    "Folder: {}".format(os.path.basename(os.path.normpath(path)))
                )
                new_items["dirs"].append(path)
            else:
                path = url.toLocalFile()
                if os.path.isfile(path):
                    self.addResponse("File: {}".format(os.path.basename(path)))
                    new_items["files"].append(path)
        self.setStatusBar(
            "Added {} files and {} directories.".format(
                len(new_items["files"]), len(new_items["dirs"])
            )
        )
        self.processNewItems(new_items)

    # --------------------------------------------------

    def closeEvent(self, event):
        """
        makes sure we close down cleanly
        """
        self.stopAndExit()
        event.ignore()

    def startThreads(self):
        global conversion_serial
        self.indicateThreadsRunning(True)
        self.setStatusBar("Working...")
        if self.queue.empty():
            self.setStatusBar(
                "{} Files converted.  You can add more, or exit.".format(
                    self.convertedCount
                )
            )
            self.indicateThreadsRunning(False)
            # self.setStatusBar("Ready.")
        try:
            while (not self.queue.empty()) and (len(self.working) < self.NUM_THREADS):
                info = self.queue.get_nowait()
                # self.addResponse('Starting conversion for "{}".'.format(info['basename']))
                conversion_serial += 1
                worker_id = conversion_serial
                worker = ConversionWorker(worker_id, info)
                thread = QThread()
                thread.setObjectName("{}".format(id))
                self.working[worker_id] = (
                    info,
                    worker,
                    thread,
                    id,
                )  # need to store worker and thread too otherwise will be gc'd
                worker.moveToThread(thread)

                # get progress messages from worker:
                worker.sig_done.connect(self.onWorkerDone)
                worker.sig_msg.connect(self.addResponse)

                # in case we need to stop the worker:
                self.sig_abort_workers.connect(worker.abort)

                # start the worker:
                thread.started.connect(worker.doConversion)
                thread.start()  # this will emit 'started' and start thread's event loop
        except queue.Empty:
            pass  # When the queue is empty, we just stop creating threads.

    @pyqtSlot(int, bool)
    def onWorkerDone(self, worker_id, error):
        thread = self.working[worker_id][2]
        # self.addResponse('Finished converting "{}" (error? {})'.format(self.working[worker_id][0]['basename'], error))
        if thread.isRunning():
            thread.quit()  # ask the thread to quit.
            thread.wait()  # <- so you need to wait for it to *actually* quit
        del self.working[worker_id]  # Now you can delete it without error.
        self.queue.task_done()
        if not error:
            self.convertedCount += 1
        if len(self.working) == 0 and self.queue.empty():
            if not self.didAbort:
                self.addResponse("[DONE]: All conversions finished.")
            else:
                self.addResponse("[STOPPED]: All conversions have been stopped.")
                self.didAbort = False
            self.setStatusBar("Ready.")
            self.indicateThreadsRunning(False)
        if not self.queue.empty():
            self.indicateThreadsRunning(True)
            self.setStatusBar("Working...")
            self.startThreads()

    @pyqtSlot()
    def abortWorkers(self):
        self.stopButton.setEnabled(False)
        self.addResponse("Stopping in-progress conversions...")
        self.didAbort = True
        with self.queue.mutex:
            self.queue.queue.clear()
            self.sig_abort_workers.emit()

    def processNewItems(self, new_items):
        converting = set()
        files = new_items["files"]
        dirs = new_items["dirs"]
        for dirname in dirs:
            if dirname is None:
                continue
            dir_files = []
            dirname = win_safe_path(dirname)
            if os.path.isdir(dirname):
                # Fancy list comprehension for folder walk from https://stackoverflow.com/a/18394205
                dir_files = [
                    os.path.join(dp, f)
                    for dp, dn, filenames in os.walk(dirname)
                    for f in filenames
                    if os.path.splitext(f)[1] in [".dcm", ".dicom", ""]
                ]
            files.extend(dir_files)
        if len(files) > 0:
            self.addResponse("Processing {} new files...".format(len(files)))
        else:
            self.addResponse("No new DICOM files were found.".format(len(files)))
        for fname in files:
            if fname is None:
                continue
            fname = win_safe_path(fname)
            if fname in converting:
                self.setResponse("[ERROR]: {} added more than once.".format(fname))
                self.abortWorkers()
                sys.exit(1)
            job_id = hashlib.sha1(
                "{}{}".format(fname, time.process_time()).encode("ascii")
            ).hexdigest()
            self.queue.put(
                {
                    "basename": os.path.basename(fname),
                    "file_path": fname,
                    "output_path": self.outputDir,
                }
            )
            converting.add(fname)

        self.startThreads()


class ConversionWorker(QObject):
    """
    Framework was modelled after this example: https://stackoverflow.com/a/41605909
    Must derive from QObject in order to emit signals, connect slots to other signals, and operate in a QThread.
    """

    sig_done = pyqtSignal(int, bool)  # worker id: emitted at end of work()
    sig_msg = pyqtSignal(str)  # message to be shown to user

    class AbortConversion(Exception):
        pass

    def __init__(self, id, info):
        super().__init__()
        self.id = id
        self.abort_requested = False
        self.task_info = info

    def checkPoint(self, info, msg=None):
        """
        See if work can continue; raises an AbortConversion exception if we need to stop.
        """
        msg_suffix = ""
        thread_id = int(QThread.currentThreadId())
        if msg is not None:
            msg_suffix = ": {}".format(msg)
        msg = 'Conversion cancelled "{}" {}'.format(info["basename"], msg_suffix)
        # check if we need to abort the loop; need to process events to receive signals;
        app.processEvents()  # this could cause change to self.abort
        if self.abort_requested:
            raise self.AbortConversion(msg)

    @pyqtSlot()
    def doConversion(self):
        """
        Performs the conversion from DICOM to PNG, applying any LUT that is embedded.
        """
        thread_id = int(QThread.currentThreadId())  # cast to int() is necessary
        error_flag = False
        # thread_name = QThread.currentThread().objectName()
        # print("Worker: id {}  name {}".format(thread_id, thread_name))

        info = self.task_info
        file_path = info["file_path"]
        output_path = info["output_path"]
        basename = info["basename"]
        self.sig_msg.emit('Starting conversion for "{}"'.format(basename, thread_id))
        try:
            self.checkPoint(info, "[0]")
            img, hdr = read_dicom(file_path)
            self.checkPoint(info, "[1]")

            if not os.path.isdir(output_path):
                os.makedirs(output_path)

            # Make a unique filename based on the DICOM header info:
            output_file = generate_unique_filename(hdr, ".pngs")
            output_file = os.path.join(output_path, output_file)

            self.checkPoint(info, "[2]")

            # Convert to greyscale in range 0-255
            img = img.astype(float)
            img -= img.min()
            img /= img.max()
            img *= 255.0

            # Convert to 8-bit unsigned int:
            img = img.astype("uint8")
            shape = img.shape

            self.checkPoint(info, "[3]")

            # Write in PNG format:
            writer = png.Writer(shape[1], shape[0], greyscale=True)
            with open(output_file, "wb") as fout:
                writer.write(fout, img)
            self.sig_msg.emit('[OK]: Finished converting "{}"'.format(basename))
        except self.AbortConversion as e:
            self.sig_msg.emit("[FAIL]: {}".format(str(e)))
            error_flag = True
        except Exception as e:
            self.sig_msg.emit('[FAIL]: Failed converting "{}" ({})'.format(basename, e))
            error_flag = True

        self.sig_done.emit(self.id, error_flag)

    def abort(self):
        if not self.abort_requested:
            self.sig_msg.emit(
                '[!]: Cancelling conversion for "{}".'.format(
                    self.task_info["basename"]
                )
            )
        self.abort_requested = True


def main():
    global app
    global conversion_serial
    conversion_serial = 0
    # install exception hook: without this, uncaught exception would cause application to exit
    # sys.excepthook = trap_exc_during_debug
    app = QtWidgets.QApplication(sys.argv)
    mainWin = ConverterWindow()
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()