#!/usr/bin/env python3
# -*- coding: utf8 -*-
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
_TEXTCOLOR = (0,0,255)
_TEXTSIZE = 0.7
_TEXFONT =  cv2.FONT_HERSHEY_SIMPLEX
_VIEWERNAME = "simpleviewer"

# For 'offline' testing 
_NOCAMERA = True

# ---- setup logging ---- 
log = logging.getLogger() # Overwrite if needed
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# ---- helper functions ---- 

def window_exists(winname):
    try:
        cv2.getWindowProperty(winname, cv2.WND_PROP_AUTOSIZE) # any property
        return True # that worked, so it exists
    except cv2.error as e:
        if e.code == -27: # (-27:Null pointer) lookup didn't find such a window
            return False
        else:
            raise # some unanticipated error
            
def update_filename(path,  add_element,  preprend = True, sep = "_"):
    """Adds to the filename path. """
    if preprend: return path.with_stem(add_element + sep + path.stem)
    else: return path.with_stem(path.stem + sep + add_element)

def nowstring(format = "%Y%m%d%H%M%S"):
    return datetime.now().strftime(format)
        
def shortID(uri_id): 
    return uri_id.split("/")[-1]
    

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
class cvCamera(abc.ABC):
    @abc.abstractmethod
    def __init__(self, videodim = (_DEFAULT_VIDEO_W, _DEFAULT_VIDEO_H),
                 stilldim = (_DEFAULT_STILL_W,  _DEFAULT_STILL_H)
                 ):
        super().__init__()
        self.frame = None # the latest image frame grabbed from the camera
        # Video format and size settings
        self.video_w,  self.video_h = videodim
        self.still_w,  self.still_h = stilldim
        # Data on the specimen identifiers found in an image
        self.lastidentifier = None
        # File handling settings
        self.dir_already_renamed = False
        self.filepath = None
        self.still_image_n = 1 # Used for output file numbering
        # Data about the image display tool
        self.viewer = None
        self.win_title = "Video stream"
        # Image processing settings
        self._zoom = 1
        self.brightness = 0 
        self.contrast = 1
        self._ZOOM_PHYSICAL = False
#        self._ZOOM_PHYS_VAL = 10
#        self._ZOOM_PHYS_STEP = 10
        self.textplace = ( 20, 20 ) # Default position
        print("self.textplace = ",self.textplace  )
        self.last_still_time = None
        self.last_save_time = None

    @abc.abstractmethod
    def frame_get(self): pass # Should return successQ, frame

    @abc.abstractmethod
    def setup(self): 
        # Set resolution
#        cv2.namedWindow(_VIEWERNAME,  cv2.WINDOW_NORMAL)
        cv2.namedWindow(_VIEWERNAME)
        cv2.createTrackbar('BRI', _VIEWERNAME, 100, 200, self.on_barchange)  # Brightness -100 to 100
        cv2.createTrackbar('CON', _VIEWERNAME,  10, 30, self.on_barchange)  # Contrast 1.0 to 3.0
        return True

    def on_barchange(*args): 
        bri = cv2.getTrackbarPos('BRI', _VIEWERNAME) - 50  #[-50, 150].
        con =  cv2.getTrackbarPos('CON', _VIEWERNAME) / 10 #[0.0, 3.0]
        args[0].contrast = con
        args[0].brightness = bri

    @abc.abstractmethod
    def shutdown(self): pass

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

    def on_still_image(self,  n):
        """Returns success status."""
	# grab raw data from the camera (with no overlay etc)
        ret, self.frame = self.frame_get()
        if not ret: return False # Could not read video	frame = 
        # Settings, should come from config file
        ATTEMPT_BARCODE = True
        RENAME_FILE_BY_TIME = True
        RENAME_FILE_BY_BARCODE = True
        RENAME_DIR_BY_BARCODE = False
        # Set up subdirectory, if it did not exist already
        if self.filepath is None: self.on_sample_done()
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
            self.last_still_time = datetime.now() # Set last save time to now
            return True
        except OSError as msg:
            log.debug(f"Saving a file triggered error {msg}")
        return False
        
    def show_frame(self):
        frame = self.frame.copy() # Should probably generate a copy here as we draw into the frame
        text = f"ID: {self.lastidentifier}"        
        text2 = ""
        if self.last_still_time and ( (datetime.now()-self.last_still_time).total_seconds() ) <1:
            text2 = f" (taking image #{self.still_image_n-1})"
        if self.last_save_time and ( (datetime.now()-self.last_save_time).total_seconds() ) < 1:
            text2 = " (saving sample)."
        text += text2
        if _DEFAULT_QR_OVERLAY and self.lastidentifier:    
            h,w = frame.shape[0:2] # Original dimensions
            self.textplace = (20, h-20)
            cv2.putText(frame, text, self.textplace,  _TEXFONT, _TEXTSIZE, _TEXTCOLOR, 2)                        
        if not self._zoom == 1: # For speed, omit scaling at no zoom.
            frame = self.zoom_digital(self._zoom)
        cv2.imshow(_VIEWERNAME, frame)
    
    def on_key(self, key):
        key_brightness_more = ord('b')
        key_brightness_less = ord('n')
        key_contrast_more = ord('c')
        key_contrast_less = ord('v')
        key_zoom_in = ord('+')
        key_zoom_out  = ord('-')
        key_sample_done = ord('s')
        key_image = ord("i")
        key_quit = ord("q")
        log.debug(f"Got command char {chr(key)}")
        if key == key_image:
            log.debug("Trying to save.")
            self.on_still_image(self.still_image_n)
            self.still_image_n += 1
        elif key == key_sample_done:
            log.debug("Saving sample")
            self.on_sample_done(mark_as_done = _DEFAULT_MARK_DONE)
        elif key == key_zoom_in: self.on_zoom(1)
        elif key == key_zoom_out:  self.on_zoom(-1)
        elif key == key_brightness_more: 
            log.debug(f"Increasing brightness from {self.brightness}"  )
            self.brightness = min(100, self.brightness+10)  
        elif key == key_brightness_less: 
            log.debug(f"Decreasing brightness from {self.brightness}"  )            
            self.brightness = max(-100, self.brightness-10) 
        elif key == key_contrast_more: self.contrast = min(3, self.contrast*1.2)  
        elif key == key_contrast_less: self.contrast = max(0, self.contrast/1.2) 
        elif key == key_quit:
            log.debug("Trying to quit")
            # Force a final newdir to make sure the last directory is treated like all the others
            self.on_sample_done(mark_as_done = _DEFAULT_MARK_DONE)
            return True
        return False
                
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
        
    def on_zoom(self,direction, step = 0.1, zoom_min = 1, zoom_max = 2):
        """Set zoom level.

        For zooming in, directory>0. For zooming out, give directory<0."""
        log.debug(f"Trying to zoom out in direction {direction}")
        if (direction > 0) and (self._ZOOM_PHYSICAL) :
            self._ZOOM_PHYS_VAL += self._ZOOM_PHYS_STEP
        elif (direction > 0): # Digital zoom
            self._zoom = min(self._zoom + step, zoom_max)
        # Zoom out
        elif (direction < 0) and (self._ZOOM_PHYSICAL) :
            self._ZOOM_PHYS_VAL -= self._ZOOM_PHYS_STEP
        else: self._zoom = max(self._zoom - step, zoom_min)

    def on_sample_done(self, mark_as_done = None):
        """Move to a new directory.  

Sets self.filepath to a timestamp-based directory, but does not create it on disk.
Return value: A Path for the new directory.
Parameters: 
- mark_as_done: if a non-empty string, write a file with this name into the completed directory.
"""
	# Create a new timestamp-based directory name
        if mark_as_done and self.filepath and self.still_image_n > 1:
            log.debug("Trying to create a marked file to sign the directory is ready for postprocessing.")                      
            try:
                donefp = self.filepath / Path(mark_as_done)            
                donefp.touch(exist_ok = True)
            except FileNotFoundError as msg:
                log.error(msg)
        self.filepath = self.basedirectory /  Path(nowstring())
        # Force creating a dir on filesystem
        self.dir_already_renamed = False
        self.last_save_time = datetime.now()
        self.still_image_n = 1
        return self.filepath
  
##    def zoom_physical(self, zoom, zoomrange = 100):
##        """Set physical zoom.
##
##        Zoom units vary between camera models. Works only with some cameras.
##        Parameters:
##        - zoomrange: range of values acceptable for zoom. Bottom is assumed to be zero."""
##        assert(zoom >= 0)
##        assert(zoom <= zoomrange)
##        self.device.set(cv2.CAP_PROP_ZOOM, zoom)

    def mainloop(self, basedirectory, namebase,  ext = "jpg",  delay=1) :                
        self.basedirectory = basedirectory
        self.namebase = namebase
        # Start looping
        log.info("Video stream started")
        while True:
            ret, self.frame = self.frame_get()
            if not ret: break # Could not read video
            # Adjust brightness  & contrast
            self.frame = cv2.convertScaleAbs(self.frame, alpha=self.contrast, beta=self.brightness)
            # Overlay identifier if available
            if _DEFAULT_QR_LIVE: 
                ids = _extract_pyzbar(self.frame)
                if len(ids) == 1: self.lastidentifier = ids[0]
            self.show_frame()
            key = cv2.waitKey(delay) & 0xFF # lowest 2
            if key == 255: pass # No nothign
            else:
                exitQ = self.on_key(key) # Should return True if an exit is needed
                if exitQ: break
            # Check if window is closed or user called for 
            if not window_exists(_VIEWERNAME): break

# --- Actual implementations ----
class StillImageVideo(cvCamera): #for testing
    def __init__(self, impath):
        super().__init__()
        self.frame = cv2.imread(impath)
        self.video_h,  self.video_w, _ = self.frame.shape
        self.textplace = ( 20,   20 )        
        log.debug(f"Frame shape is {self.video_w}x{self.video_h}")
    def setup(self): 
        rv = super().setup()
        if not rv: return rv # Super setup failed
        self._frame_data = cv2.imread('./test.jpg')
        return True
    def shutdown(self):
        cv2.destroyAllWindows() # Close all cv2 windows. Has no return value :(
        return True       
    def frame_get(self): 
        """Return True, framedata (mirroring cv2 behavior)."""
        return (True,  self._frame_data)
    def on_still_image(self, n): 
        self.last_still_time = datetime.now() # Set last save time to now
    def on_sample_done(self, mark_as_done = None): 
        self.last_save_time = datetime.now() # Set last save time to now
        self.still_image_n = 1
        

class WebCamVideo(cvCamera):
    """Video (and still from video) capture. Based on cv2."""
    def __init__(self, videodev = _DEFAULT_VIDEO_SRC, \
                 videodim = (_DEFAULT_VIDEO_W, _DEFAULT_VIDEO_H),
                 stilldim = (_DEFAULT_STILL_W,  _DEFAULT_STILL_H)
                 ):
        super().__init__(videodim,  stilldim)
        self.device_address = videodev
        self.device = None
        
    def setup(self):
        """Set up video device for capture"""
        rv = super().setup()
        if not rv: return rv # Super setup failed
        self.device = cv2.VideoCapture(self.device_address)
        self.device.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_w)
        self.device.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_h)        
        if ( self.device is None ) or ( not self.device.isOpened() ): return False    
        else: return True # Success
        
    def shutdown(self):
        if self.device: self.device.release()		
        cv2.destroyAllWindows() # Close all cv2 windows. Has no return value :(
        return True       

    def frame_get(self):  return self.device.read()

# ----- MAIN SCRIPT ---- 
if __name__ == '__main__': 
    # Test video capture
    if _NOCAMERA: videodevice = StillImageVideo("test.jpg")
    else: videodevice = WebCamVideo()
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
            
# TODO: MOVE CAMERA SETUP TO TOML 
# TODO: Store video option?
