import cv2
import sys
import time
import glob
import os

import imutils

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


version = '20200822' # first working vesion, tracking a mess, properties are only estimated for first occurence
version = '20201120' # trying different threshold self.minArea = 100 self.minMaxMinRatio = 0.1


'''
To do:

* change tracking concept, detect everything, then see whether ther eis a matc
with previous particles also considering speed and direction if multiple images 
are available

* add flag for particles at the edge

'''
# https://pypi.org/project/memory-profiler/

'''
identify interesting time stamps
'''



def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: 
        return None
    else:
        return (x, y, w, h)


class particles(object):
    def __init__(self,
                fnameOut,
                 history=500,
                 dist2Threshold=400.0,
                 detectShadows=False,
                 showResult=True,
                 verbosity=10,
                 composite = True
                 ):

        self.verbosity = verbosity
        self.showResult = showResult
        self.composite = composite
        self.minArea = 100
        self.minMaxMinRatio = 0.1

        self.all = {}
        self.active = {}
        self.pp = 0
        self.fgMask = None

        self.nMovingPix = 0
        self.blur = 0
        self.fid = 0

        self.fnameOut = fnameOut

        self.backSub = cv2.createBackgroundSubtractorKNN(
            history=history,
            dist2Threshold=dist2Threshold,
            detectShadows=detectShadows
        )

        return

    def update(self, frame, minMovingPix=500, minBlur=20):

        if (self.verbosity > 2):
            print("particles.update", "FRAME", self.fid,  'Start %s' % 'update')
        self.frame = frame
        self.frame4drawing = frame.copy()

        if self.N > 0:
            self.updateTracking()

        self.updateBackground()
        if (self.fid > 5) and (self.nMovingPix > minMovingPix) and (self.blur > minBlur):
            self.updateDetection()



        else:
            if (self.verbosity > 3):
                print("particles.update", "FRAME", self.fid, 'all deactivated, blur & movingPix threshold')
            self.deactivateAll()

            if (self.verbosity > 3):
                print("particles.update", "FRAME", self.fid, 'Skipped updateDetection %i %f' %
                  (self.nMovingPix, self.blur))

        self.fid += 1
 
    def updateBackground(self):
        if (self.verbosity > 2):
            print("particles.updateBackground", "FRAME", self.fid, 'Start %s' % 'updateBackground')
        self.fgMask = self.backSub.apply(self.frame)
        self.nMovingPix = self.fgMask.sum()/255
        blurMap = cv2.Laplacian(self.frame, cv2.CV_16S)
        self.blur = blurMap[self.fgMask != 0].var()

    def updateDetection(self):
        if (self.verbosity > 2):
            print("particles.updateDetection", "FRAME", self.fid, 'Start %s' % 'updateDetection')
        thresh = cv2.dilate(self.fgMask, None, iterations=2)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        print("particles.updateDetection", "FRAME", self.fid, 'updateDetection found', len(cnts), 'particles')

        if len(cnts) == 0:
            print("particles.updateDetection", "FRAME", self.fid, 'all deactivated, no particles found')
            self.deactivateAll()


        # loop over the contours
        for cnt in cnts:


            added, particle1 = self.add(self.frame, cnt, self.fnameOut, verbosity=self.verbosity, composite = self.composite)
            if added:
                print("particles.updateDetection", "PID", particle1.pid, "Added %i area"%(particle1.area))
                if self.showResult:
                    particle1.drawContour(self.frame4drawing)
                    particle1.annotate(self.frame4drawing, extra='detected')
            else:
                print("particles.updateDetection", "PID", particle1.pid, "Not added")

    def updateTracking(self):
        if (self.verbosity > 2):
            print("particles.updateTracking", "FRAME", self.fid, 'Start %s' % 'updateTracking')

        for pid, particle1 in list(self.active.items()):
            if self.fid == particle1.fid1:
                print("particles.updateTracking", "PID", particle1.pid, "Don't track first")
                continue
            ok = particle1.update(self.frame)
            if not ok:
                print("particles.updateTracking", "PID", particle1.pid, "Tracker lost")
                particle1.annotate(self.frame4drawing, extra='LOST', color=(0, 0, 255))
                self.deactivate(pid)
            elif self.showResult:
                print("particles.updateTracking", "PID", particle1.pid, "Tracking works")
                particle1.drawContour(self.frame4drawing, color=(255, 0, 0))
                particle1.annotate(self.frame4drawing, extra='tracking')

    def add(self, *args, **kwargs):
        particleAdd = particle(self.fid, *args, **kwargs)
        newParticle = True
        added = False


        if particleAdd.area < self.minArea:
            print("particles.add", "PID", particleAdd.pid, "too small", particleAdd.area)
            return added, particleAdd

        if particleAdd.maxVal/particleAdd.minVal < self.minMaxMinRatio:
            print("particles.add", "PID", particleAdd.pid, "too small minMaxMinRatio", particleAdd.area)
            return added, particleAdd

        particleFrame = particleAdd.currentImage
        if np.prod(particleFrame.shape) == 0:
            print("particles.add", "PID", particleAdd.pid, "skipped", particleFrame.shape)
            return added, particleAdd

        print("FILTER", particleFrame.mean(), particleFrame.max()   , particleFrame.std()   )

        # plt.figure()
        # plt.title(particleAdd.pid)
        # plt.imshow(particleFrame)
        # plt.show()
        for pid, particleAactive in list(self.active.items()):
            inters = intersection(particleAdd.roi, particleAactive.roi)
            print("particles.add", "PID", pid, "intersection", inters)
            if inters is not None:
                newParticle = False
        if newParticle:
            particleAdd.startTracking(self.pp, self.frame)
            self.all[self.pp] = particleAdd
            self.active[self.pp] = particleAdd
            print("particles.add", "PID", particleAdd.pid, "Added")
            self.pp += 1
            added = True
        else:
            print("particles.add", "PID", particleAdd.pid, "Exists")
            
        self.lastParticle = particleAdd

        return added, particleAdd

    def deactivate(self, pid):
        if (self.verbosity > 2):
            print("particles.deactivate", "PID", pid, "Deactivated")
        self.all[pid].createComposite()
        del self.all[pid].tracker
        del self.all[pid].images
        del self.active[pid]

        return self.all[pid]

    def deactivateAll(self):
        if (self.verbosity > 2):
            print("particles.deactivateAll", "PID", "ALL", "Deactivated")
        for pid in self.active.keys():
            self.all[pid].createComposite()
            del self.all[pid].tracker    
            del self.all[pid].images    
        self.active = {}

    def collectResults(self):
        self.particleProps = xr.Dataset(coords = {'pid':range(len(self.all))}) 
        for key in ['Dmax', 'Dmin', 'area', 'aspectRatio', 'angle', 'Dx', 'Dy', 'x', 'y', 'perimeter', 'minVal', 'maxVal', 'blur', 'fid1']:
            arr = []
            for i in self.all.values():
                arr.append(getattr(i, key))
            self.particleProps[key] = xr.DataArray(arr,coords=[self.particleProps.pid])
        return self.particleProps



    @property
    def N(self):
        return len(self.active)

    @property
    def pids(self):
        return list(self.active.keys())


class particle(object):
# TrackerKCF_create
    def __init__(self, fid1, frame1, cnt, fnameOut, verbosity=0, composite = True):
        self.verbosity = verbosity
        #start with negative random id
        self.pid = np.random.randint(-999,-1) 
        self.fid1 = fid1
        self.cnt = cnt
        self.fnameOut = fnameOut
        self.composite = composite

        self.first = True
        self.lost = False


        self.images = []
        self.rois = []

        self.estimateProperties(frame1)

        return

    def __repr__(self):
        props = "#"*30
        props += '\n'
        props += 'PID: %i\n'%self.pid
        props += 'Dmax, Dmin: %i %i\n'%(self.Dmax, self.Dmin)
        props += 'aspectRatio: %f\n'%(self.aspectRatio)
        props += 'angle: %f\n'%(self.angle)
        props += 'Dx, Dy: %i %i\n'%(self.Dx, self.Dy)
        props += 'Area: %i\n'%self.area
        props += 'Centroid: %i, %i\n'%self.centroid
        props += 'Perimeter: %i\n'%self.perimeter
        props += 'minVal/maxVal: %i %i %f\n'%(self.minVal, self.maxVal, self.minVal/self.maxVal)
        props += 'Blur: %f\n'%self.blur

        return props

    @property
    def NSamples(self):
        return len(self.images)
    

    def extractRoi(self, frame):
        x, y, w, h = self.roi
        if len(frame.shape) == 3:
            frame = frame[:,:,0]
        return frame[y:y+h, x:x+w]


    def estimateProperties(self, frame):

        self.roi = tuple(int(b) for b in cv2.boundingRect(self.cnt))
        x, y, self.Dx, self.Dy = self.roi
        if self.verbosity > 1:
            print("particle.__init__", "PID", self.pid, 'found particle at %i,%i, %i, %i' % self.roi)

        self.currentImage = self.extractRoi(frame)
        self.minVal = self.currentImage.min()
        self.maxVal = self.currentImage.max()

        self.rect = cv2.minAreaRect(self.cnt)
        #todo: consider https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
        center, dims, self.angle = self.rect
        self.Dmax = max(dims)
        self.Dmin = min(dims)
        self.aspectRatio = self.Dmin / self.Dmax

        self.area = cv2.contourArea(self.cnt)
        M = cv2.moments(self.cnt)
        #https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
        self.centroid = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        self.x, self.y = self.centroid
        self.perimeter = cv2.arcLength(self.cnt,True)

        self.blur = cv2.Laplacian(frame, cv2.CV_16S).var()


        self.rois.append(self.roi)
        self.images.append(self.currentImage)


    def startTracking(self, pid, frame1, tracker=cv2.TrackerTLD_create):

        self.pid = pid
        self.tracker = tracker()
        ok = self.tracker.init(frame1, self.roi)
        if self.verbosity > 2:
            print("particle.startTracking", "PID", self.pid, 'added particle at %i,%i' % (self.x, self.y))


    def drawContour(self, frame, color=(0, 255, 0)):
        assert not self.lost
        (x, y, w, h) = self.roi
        cv2.rectangle(frame, (x, y),
                      (x + w, y + h), color, 2)

        if self.NSamples == 1:
            cv2.drawContours(frame, [self.cnt],0,np.array(color) * 2/3,2)
            box = cv2.boxPoints(self.rect)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,np.array(color) * 1/3,2)

        return frame

    def annotate(self, frame, color=(0, 255, 0), extra=''):
        cv2.putText(frame, '%i %s' % (self.pid, extra),
                    (self.x, self.y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


    def update(self, frame):
        assert self.pid >= 0
        ok, roi = self.tracker.update(frame)
        if ok:
            self.roi = tuple(int(b) for b in roi)
            (self.x, self.y, w, h) = self.roi
            if self.verbosity > 3:
                print("particle.update", "PID", self.pid, "Tracker works, now at", self.x, self.y)
            self.first = False

            self.currentImage = self.extractRoi(frame)

            self.images.append(self.currentImage)
            self.rois.append(self.roi)

        else:
            if self.verbosity > 3:
                print("particle.update", "PID", self.pid, "Tracker lost")
            self.lost = True
        return ok

    def createComposite(self):
        if self.composite:
        
            maxH = max([np.shape(a)[0] for a in self.images]) 
            images = []
            for image in self.images:
                if np.prod(image.shape) == 0:
                    continue
                hh = image.shape[0]
                h1 = int(np.floor((maxH - hh)/2))
                h2 = int(np.ceil((maxH - hh)/2))  
                images.append(np.pad(image, ((h1, h2), (0,0)), 'constant', constant_values=255))
                images.append(np.zeros((maxH, 1)))
            self.compositeImage = np.hstack(images)

            cv2.imwrite("%sparticle_v%s_%i.png"%(self.fnameOut, version, self.pid), self.compositeImage)

            return self.compositeImage
        else:
            return None


