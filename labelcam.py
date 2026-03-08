"""A tool for capturing still images from a video stream, storing them into a directory

Intended for label image capturing and feeding an AI for data extraction.
"""
import abc, logging, sys
from datetime import datetime
from pathlib import Path
from pyzbar import pyzbar
import cv2

_DEFAULT_VIDEO_SRC = 0 # 1st video source
_DEFAULT_VIDEO_W = 1920
_DEFAULT_VIDEO_H = 1080
_DEFAULT_STILL_W = 1920
_DEFAULT_STILL_H = 1080
_KEEP_ORIGINAL_DIM_ON_ZOOM = True
_DEFAULT_MARK_DONE = "imaging_done"
_DEFAULT_QR_OVERLAY = True	
_DEFAULT_QR_LIVE = True	
_TEXTCOLOR = (0,0 ,0)

# ---- setup logging ---- 
log = logging.getLogger() # Overwrite if needed
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# ---- helper functions ---- 
def update_filename(path,  add_element,  preprend = True, sep = "_"):
    """Adds to the filename path. """
    if preprend: return path.with_stem(add_element + sep + path.stem)
    else: return path.with_stem(path.stem + sep + add_element)

def nowstring(format = "%Y%m%d%H%M%S"):
    return datetime.now().strftime(format)
        
def shortID(uri_id): 
    return uri_id.split("/")[-1]
    
def display(img, title=""):
    """Display an image using cv2. Mostly for testing"""
    cv2.imshow(title, img)    
    cv2.waitKey(0)         # wait for user to press a key
    cv2.destroyAllWindows() # Close all cv2 windows

def shrink_to_maxdim(img,maxdim):
    md = max(img.shape)
    scale = min(1.0, maxdim/md)
    return cv2.resize(img,(0,0),fx=scale,fy=scale)
    
def _extract_pyzbar(image, encoding="utf8",  maxsize = 640):
    barcodes = []
    # try qr recognition at different image sizes
    if (maxsize < 640): sizes = (maxsize,) # Try just one size
    else:sizes = (320,640, min(2000, max(image.shape)) ) # Caps largest try size to 2000 for speed
    for maxdim in sizes:
        smallimg = shrink_to_maxdim(image,maxdim)
        barcodes = pyzbar.decode(smallimg, symbols=[pyzbar.ZBarSymbol.QRCODE])
        if barcodes or maxdim == max(image.shape): break
    d = []
    for qr in barcodes:
        bkd = qr.data
        if encoding: bkd = bkd.decode(encoding) 
        d.append(bkd)
    return [x for x in d if x] # Make sure the result is a list of non-empty string
    
# ---- Classes for storing and processing images ----
class Labelmage():
    def __init__(self,  img):
        self.img = img
        self._identifier = None # property
    @property
    def identifier(self): return self._identifier
    @identifier.setter
    def identifier(self, val): self._identifier = val
    
    def save(self,  path):
        # TODO: Test if path exists, writeable etc.?
        if not path: return False # Fail to save
        return cv2.imwrite(str(path), self.img) #cv2 does not understand Paths        

    def barcodestrings(self):
        """"Find all barcode data in image. 
        
        Return value may contain duplicate values"""
        return _extract_pyzbar(self.img)


# ---- Classes for operating cameras ----
class Camera(abc.ABC):
    def __init__(self):
        self.device = None
    @abc.abstractmethod
    def setup(self): pass
    @abc.abstractmethod
    def shutdown(self): pass

class WebCamVideo(Camera):
    """Video (and still from video) capture. Based on cv2."""
    def __init__(self, videodev = _DEFAULT_VIDEO_SRC, \
                 videodim = (_DEFAULT_VIDEO_W, _DEFAULT_VIDEO_H),
                 stilldim = (_DEFAULT_STILL_W,  _DEFAULT_STILL_H)
                 ):
        super().__init__()
        self.device_address = videodev
        self.lastidentifier = None
        self.device = None
        self.win_title = "Video stream"
        self.video_w,  self.video_h = videodim
        self.still_w,  self.still_h = stilldim
        self.frame = None
        self.dir_already_renamed = False
        self.filepath = None
        self._zoom = 1
        self._ZOOM_PHYSICAL = False
        self._ZOOM_PHYS_VAL = 10
        self._ZOOM_PHYS_STEP = 10
        
        # Todo: set all variables to default values here!
        
    def zoom_digital(self, zoom):
        """Digital zoom. Soom range: [1-2]"""
        h,w = self.frame.shape[0:2] # Original dimensions
        newh, neww = [ int(x/zoom) for x in (h,w) ]
        top =  int( (h-newh)/2 )
        left=  int( (w-neww)/2 )
        cropped = self.frame[top:top+newh, left:left+neww]
        if _KEEP_ORIGINAL_DIM_ON_ZOOM:
            return cv2.resize(cropped, (w,h)) # Scale to original dimensions
        else:
            return cropped

##    def zoom_physical(self, zoom, zoomrange = 100):
##        """Set physical zoom.
##
##        Zoom units vary between camera models. Works only with some cameras.
##        Parameters:
##        - zoomrange: range of values acceptable for zoom. Bottom is assumed to be zero."""
##        assert(zoom >= 0)
##        assert(zoom <= zoomrange)
##        self.device.set(cv2.CAP_PROP_ZOOM, zoom)

    def setup(self):
        """Set up video device for capture"""
        self.device = cv2.VideoCapture(self.device_address)
        # Set resolution
        self.device.set(cv2.CAP_PROP_FRAME_WIDTH, self.still_w)
        self.device.set(cv2.CAP_PROP_FRAME_HEIGHT, self.still_h)        
        if ( self.device is None ) or ( not self.device.isOpened() ): return False    
        else: return True # Success
        
    def shutdown(self):
        if self.device: self.device.release()		
        cv2.destroyAllWindows() # Close all cv2 windows. Has no return value :(
        return True       
        
    def on_zoom(self,direction, step = 0.1, zoom_min = 1, zoom_max = 2):
        """Set zoom level.

        For zooming in, directory>0. For zooming out, give directory<0."""
        if (direction > 0) and (self._ZOOM_PHYSICAL) :
            self._ZOOM_PHYS_VAL += self._ZOOM_PHYS_STEP
        elif (direction > 0): # Digital zoom
            self._zoom = min(self._zoom + step, zoom_max)
        # Zoom out
        elif (direction < 0) and (self._ZOOM_PHYSICAL) :
            self._ZOOM_PHYS_VAL -= self._ZOOM_PHYS_STEP
        else: self._zoom = max(self._zoom - step, zoom_min)

    def on_newdir(self, mark_as_done = None):
        """Move to a new directory.  

Sets self.filepath to a timestamp-based directory, but does not create it on disk.
Return value: A Path for the new directory.
Parameters: 
- mark_as_done: if a non-empty string, write a file with this name into the completed directory.
"""
	# Create a new timestamp-based directory name
        if mark_as_done and self.filepath:
            log.debug("Trying to create a marked file to sign the directory is ready for postprocessing.")                      
            donefp = self.filepath / Path(mark_as_done)
            donefp.touch(exist_ok = True)
        self.filepath = self.basedirectory /  Path(nowstring())
        # Force creating a dir on filesystem
        self.dir_already_renamed = False
        return self.filepath

    def save_image(self,sampleimage, filename):
        fullpath = self.filepath / filename
        log.debug(f"Trying to save image in {fullpath}")
        if not self.filepath.exists(): self.filepath.mkdir()        
        if sampleimage.save(fullpath): log.debug(f"Saved image in {self.filepath}") 
        else: log.debug(f"Failed to save image in {fullpath}")         

    def save_identifier(self,sampleimage, filename):
        fullpath = self.filepath / filename
        log.debug(f"Trying to save identifer in {fullpath}")
        if not self.filepath.exists(): self.filepath.mkdir()
        try:
            with fullpath.open("w") as f:
                f.write(sampleimage.identifier or "") # Empty file if no identifier
            log.debug(f"Saved identifer in {self.filepath}")
        except OSError as msg:
            log.error(f"Failed to save identifer in {fullpath}: {msg}")         

    def on_save(self,  n):
        # Settings, should come from config file
        ATTEMPT_BARCODE = True
        RENAME_FILE_BY_TIME = True
        RENAME_FILE_BY_BARCODE = True
        RENAME_DIR_BY_BARCODE = False
        # Set up subdirectory, if it did not exist already
        if self.filepath is None: self.on_newdir()
        # Create LabelImage object
        sampleimage = Labelmage(self.frame)
        filename= Path(f"{self.namebase}_{n}.{ext}")
        self.filepath = self.filepath # Default file name
        # Reset file name for the image if needed
        if RENAME_FILE_BY_TIME:                     
            filename = update_filename(filename,  nowstring() ,  preprend = True, sep = "_")             
            log.debug(f"New path after file rename {self.filepath}.")
        # Attempt barcode recognition
        if ATTEMPT_BARCODE: 
            # All the logic below should probably be in a try block  rather than ifs
            log.debug("Trying barcode detection.")
            ids = sampleimage.barcodestrings()
            log.debug(f"Barcode detection result: {ids}.")
            if  len(ids) == 1: # Just 1 barcode detected
                self.lastidentifier = ids[0]
                shortid = shortID(self.lastidentifier)
                if RENAME_FILE_BY_BARCODE: 
                    log.debug("File rename by barcode content called")
                    filename = update_filename(filename,  shortid,  preprend = True, sep = "_")                    
                    log.debug(f"new path after bk rename is {self.filepath}" )
                    log.debug("File rename done")
                if RENAME_DIR_BY_BARCODE and not self.dir_already_renamed: 
                    self.dir_already_renamed = True # Set to true even if the attempt failed
                    self.filepath =  self.filepath.with_stem(shortid)
                    # TODO: if the firectory exists, should rename!
                    log.debug(f"New dir is {self.filepath}" )
        try:
            self.save_image(sampleimage,filename)
            self.save_identifier(sampleimage,filename.with_suffix(".identifier"))
        except OSError as msg:
            log.debug(f"Saving a file triggered error {msg}")
        
    def show_frame(self):
        if _DEFAULT_QR_OVERLAY and self.lastidentifier:             
            cv2.putText(self.frame, f"ID: {self.lastidentifier}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, _TEXTCOLOR, 2)            
        if self._zoom == 1: # For speed, omit scaling at no zoom.
            cv2.imshow(self.win_title, self.frame)
        else:
            self.frame = self.zoom_digital(self._zoom)
            cv2.imshow(self.win_title, self.frame)
        
    def mainloop(self, basedirectory, namebase, \
            ext = "jpg",  delay=1,  save_key = "s", quit_key = "q",  new_dir_key = "n"):                
        self.basedirectory = basedirectory
        self.namebase = namebase
        # Start looping
        log.info("Video stream started")
        log.info(f"{save_key} to save,  {quit_key} to quit")
        still_image_n = 1 # Used for output file numbering
#        self.device.set(cv2.CAP_PROP_BRIGHTNESS, 100)
        while True:
            ret, self.frame = self.device.read()
            if not ret: break # Could not read video
            # Overlay identifier if available
            if _DEFAULT_QR_LIVE: 
                ids = _extract_pyzbar(self.frame)
                if len(ids) == 1: self.lastidentifier = ids[0]
            self.show_frame()
            key = cv2.waitKey(delay) & 0xFF
            if key == ord(save_key):
                log.debug("Trying to save.")
                self.on_save(still_image_n)
                still_image_n += 1
            elif key == ord(new_dir_key):
                log.debug("Trying to move into a new directory")
                self.on_newdir(mark_as_done = _DEFAULT_MARK_DONE)
            elif key == ord("+"):
                log.debug("Trying to zoom in")
                self.on_zoom(1)
            elif key == ord("-"):
                log.debug("Trying to zoom out")
                self.on_zoom(-1)
            elif key == ord(quit_key):
                log.debug("Trying to quit")
                # Force a final newdir to make sure the last directory is treated like all the others
                self.on_newdir(mark_as_done = _DEFAULT_MARK_DONE)
                break
            
if __name__ == '__main__': 
    # Test video capture
    videodevice = WebCamVideo()
    if not videodevice.setup():
        log.error("Opening video device failed")
        sys.exit()
    videopath = Path("/home/kahanpaa/labelwebcam/data")
    if not videopath.exists(): videopath.mkdir()
    basename = "sample"
    ext = "jpg"
    videodevice.mainloop(videopath,basename,ext)


import atexit
@atexit.register
def onexit():    
    if "videodevice" in globals():  # Try to close video device cleanly
        log.debug("Shutting down video device")
        if not videodevice.shutdown(): log.warning("Properly closing video device failed")
    log.info("done")
            
