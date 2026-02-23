import numpy as np
import matplotlib.pylab as plt
import h5py as h5
import silx.io.h5py_utils
from lmfit.models import RectangleModel, GaussianModel, QuadraticModel
from scipy.interpolate import interp1d, pchip_interpolate
try:
    from scipy.interpolate import make_smoothing_spline
except ImportError:
    pass


from scipy.signal import savgol_filter
from scipy.signal import find_peaks as fp
#from skimage.restoration import rolling_ball 

from numpy import rad2deg as deg
from numpy import deg2rad as rad
import os
import datetime

try :
    import spec_reader as sr
except ImportError:
    print("No spec_reader module installed, cannot read legacy SPEC data.")
    print("... probably OK if you are looking at recent data.")

class Sample():
    def __init__(self, sampleName, datafilePathXRR, scanNbrXRR, 
                 datafilePathBG=None, scanNbrBG=None, BG_subrange=None, sampleLength=4, darkCurrent=0, 
                 directBeamFHWM = 0.0033, wavelength = 12398/27000,
                 directBeamAmplitude=None, beamSize=0.05, tthOffset=0,
                 zrange=100, qmax=None,
                 tthmin=0.05, tthmin_bg=0.1,
                 qvaluesRangesToRemove = [],
                 peak_finding_method = "scipy",
                 peak_distance=None,peak_prominence=None,
                 baseline_type="pchip",
                 peak_distance_scipy=30,
                 peak_prominence_scipy=1E-10):
        self.sampleName = sampleName
        self.datafilePathXRR = datafilePathXRR
        self.basefilename = os.path.split(datafilePathXRR)[-1].split(".h5")[0]
        ##test
        self.scanNbrXRR = scanNbrXRR
        self.scanNbrBG = scanNbrBG
        self.peak_distance = peak_distance
        self.peak_prominence = peak_prominence
        self.sampleLength = sampleLength
        self.darkCurrent = darkCurrent
        self.directBeamFHWM = directBeamFHWM
        self.wavelength = wavelength
        self.directBeamAmplitude=directBeamAmplitude
        self.tthOffset=tthOffset
        self.beamSize = beamSize
        self.zrange = zrange
        self.qmax = qmax
        self.tthmin = tthmin
        self.tthmin_bg = tthmin_bg
        self.BG_subrange = BG_subrange
        self.max_idx = None
        self.peaks = [] 
        self.minima = []
        self.baseline_type = baseline_type
        self.peak_finding_method = peak_finding_method
        self.peak_distance_scipy = peak_distance_scipy
        self.peak_prominence_scipy = peak_prominence_scipy
        self.qvaluesRangesToRemove  = qvaluesRangesToRemove
        if datafilePathBG == None:
            self.datafilePathBG = datafilePathXRR
        else:
            self.datafilePathBG = datafilePathBG

        # Valeur des coeff d’attenuation pour Cu : attnfactlist[i] donne la valeur de i attenuateurs
        self.attnFactors27keV = np.array([1.00000000e+00, 2.66000000e+00, 7.02000000e+00, 1.86732000e+01,
                                          4.92000000e+01, 1.30872000e+02, 3.45384000e+02, 9.18721440e+02,
                                          2.42000000e+03, 6.43720000e+03, 1.69884000e+04, 4.51891440e+04,
                                          1.19064000e+05, 3.16710240e+05, 8.35829280e+05, 2.22330588e+06])
        
    def process(self):
        # read the data to set the values for:
        # self.tth, self.det, self.mon, self.integrationTime
        # self.tth_bg, self.det_bg, self.mon_bg, self.integrationTime_bg
        self.get_rawdata()
        
        # do the different corrections and set the values for:
        # self.mondc, self.detm, self.tthc, self.detmn, self.fpc, self.detmnp, self.imin,self.imax, self.tthcr, self.detmnpr
        # self.qraw, self.Raw
        self.preprocess()
        
        # limit to positive R values only and set the values for:
        # self.qc, self.Rc
        self.ensure_positivity()
        
        # limit to qmax (if needed) and set the values for:
        # self.q, self.R
        self.crop()

        # further remove (possibly) q-ranges
        self.remove_qvalues()
        
        

    def inverse(self):
        from scipy.integrate import cumtrapz
        ## process the raw data into q, R 
        self.process()
        ## invert the sign of R*q**4 for every adjacent peak without/with baseline subtraction
        self.sqrtRq4,self.sqrtRq4_bl,self.sqrt_smoothed_Rq4_bl = self.invert_bl(self.R*self.q**4)
        ## do the inverse FT and integrate, and set the values for:
        ## self.z,self.deltaRho
        self.d_rho_nc, self.z_nc = self.IFT_slow(np.sqrt(self.R*self.q**4),self.q,zrange=self.zrange)
        self.deltaRho_nc = cumtrapz(self.d_rho_nc.imag,self.z_nc,initial=0)
        #self.z_nc, self.deltaRho_nc = self.ift(self.q, np.sqrt(self.R*self.q**4))
        ## self.z, self.deltaRho 
        self.d_rho, self.z = self.IFT_slow(self.sqrtRq4,self.q,zrange=self.zrange)
        self.deltaRho = cumtrapz(self.d_rho.imag,self.z,initial=0)       
        #self.z, self.deltaRho = self.ift(self.q, self.sqrtRq4)
        ## self.z_bl, self.deltaRho_bl        
        self.d_rho_bl, self.z_bl = self.IFT_slow(self.sqrtRq4_bl,self.q,zrange=self.zrange)
        self.deltaRho_bl = cumtrapz(self.d_rho_bl.imag,self.z_bl,initial=0)     
        #self.z_bl, self.deltaRho_bl = self.ift(self.q, self.sqrtRq4_bl)
        ## self.z_bl_smoothed, self.deltaRho_bl_smoothed      
        self.d_rho_bl_s, self.z_bl_s = self.IFT_slow(self.sqrt_smoothed_Rq4_bl,self.q,zrange=self.zrange)
        self.deltaRho_bl_s = cumtrapz(self.d_rho_bl_s.imag,self.z_bl_s,initial=0)       
        #self.z_bl_s, self.deltaRho_bl_s = self.ift(self.q, self.sqrt_smoothed_Rq4_bl)

        
    def smooth(self,y,win_size=None):
        # Estimate window size based on curve characteristics
        curve_range = np.max(y) - np.min(y)
        if win_size != None:
            window_size = win_size
        else:
            window_size = int(0.05 * len(y) + 0.5 * curve_range)
        self.window_size = window_size
        poly_order = 3
        #ensure window_size > poly_order
        if window_size <= poly_order:
            window_size=poly_order+1
        #window_size must be odd
        if window_size%2 == 0:
            window_size +=1
        #smoothed_y = savgol_filter(y, window_size*2+1, poly_order)
        smoothed_y = savgol_filter(y, window_size, poly_order)
        return smoothed_y

    
    def find_peaks(self,y,period,prominence):
        #find the peaks
        # Initialize variables
        peaks = []
        max_idx = np.argmax(y)
        peaks.append(max_idx)
        last_peak_index = max_idx
        # Iterate through the data to find peak
        #for the left side of the max peak
        for i in range(max_idx+1,0,-1):
            if i-int(period/2)>0:
                sub_range=y[i-int(period/2):i]
            else:
                sub_range=y[0:i]
            local_max=np.max(sub_range)

            if y[i] > y[i - 1] and y[i] > y[i + 1]:
            # Check if the current index is consistent with the known peak distance
                if np.abs(i - last_peak_index) >= int(0.7*period) and y[i]>=local_max:
                    peaks.append(i)
                    last_peak_index = i
        #for the right side of the max peak   
        last_peak_index=max_idx
        for i in range(max_idx+1,len(y)-1):
            if i+int(period/2)<len(y):
                sub_range=y[i:i+int(period/2)]
            else:
                sub_range=y[i:len(y)]
            local_max=np.max(sub_range)
            if y[i] > y[i - 1] and y[i] > y[i + 1]:
            # Check if the current index is consistent with the known peak distance
                if np.abs(i - last_peak_index) >= int(0.7*period) and y[i]>=local_max:
                    peaks.append(i)
                    last_peak_index = i
        peaks.sort()
        self.peaks=peaks


    def find_min(self,y,period,prominence,peaks):
        #peaks=[]
        #peaks=self.peaks
        #find minima of the curve Rq4 vs q
        # Initialize variables
        minima = []
        last_minima_index = 0
        #min_idx=0
        # Iterate through the data to find minima
        for i in range(peaks[0]+1, len(y) - 1):
            end_idx=np.min([i+6,i+int(period/4),len(y)-1])
            sub_range=y[i:end_idx]
            local_min=np.min(sub_range)
            if y[i] < y[i - 1] and y[i] < y[i + 1]:
            # Check if the current index is consistent with the known minima distance
                if i - last_minima_index >= 0.6*period :
                    if y[i]-np.min(y)<0.25*prominence and y[i] <= local_min:
                        minima.append(i)
                        last_minima_index = i
                        #min_idx +=1
        minima.sort()
        self.minima=minima
    

    def find_min2(self,y,period,prominence,peaks):
        p,_ = fp(-y, distance=4, prominence=1E-10)
        self.minima=p
    

    #def find_min_scipy(self, y):
    #    # first pass, find the most obvious minima
    #    p,_ = fp(-y, distance=self.peak_distance_scipy, prominence=self.peak_prominence_scipy)
    #    # remove the background with a rolling ball of radius = position of first peak
    #    radius = p[0]
    #    intensity_scale_factor = radius/y.max()
    #    bg = rolling_ball(y*intensity_scale_factor, radius=radius)/intensity_scale_factor
    #    # second pass, find the minima with the background removed
    #    if len(p)>1 :
    #        p2,_ = fp(-(y-bg), distance=(p[1]-p[0])*.75)
    #    else:
    #        p2,_ = fp(-(y-bg), distance=p[1]*.75)
    #    self.minima = p2



    def invert_bl(self,y):
        ### enter y, return sqrt_y,sqrt_y_bl,sqrt_smoothed_y_bl
        sqrt_y=np.sqrt(y)
        #smooth the curve
        smoothed_y=sqrt_y #self.smooth(y,win_size=None)
        self.smoothed_Rq4=smoothed_y

        if 0:#self.peak_finding_method == "scipy":
            #print("here")
            #self.find_min_scipy(smoothed_y)
            pass
        else:
        #define period and prominence
            max_idx = np.argmax(y[:len(y)//2+1])
            prominence = y[max_idx]-np.min(y)
            self.prominence = prominence
            min_next_to_max = 0
            for i in range(max_idx,len(y)-1):
                if y[i]<y[i+1] and y[i]<y[i-1] and y[max_idx]-y[i]>=0.9*prominence:
                    min_next_to_max=i
                    break
            self.max_idx=max_idx
            self.min_next_to_max=min_next_to_max
            #Initialize period
            period=int(2*(min_next_to_max-max_idx))
            self.period = period
            #find peaks and minima 
            self.find_peaks(smoothed_y,period=period,prominence=prominence)
            #self.minima=self.find_peaks(-smoothed_y,period=period,prominence=prominence)
            #self.find_min(smoothed_y,period=period,prominence=prominence,peaks=self.peaks)
            self.find_min2(smoothed_y,period=period,prominence=prominence,peaks=self.peaks)
        
        # Create an interpolation function to draw the baseline
        if self.baseline_type == "linear":
            if len(self.minima) < 2:
                baseline_func = lambda x: np.zeros_like(x)  # Define a baseline function that returns an array of zeros
            else:
                baseline_func = interp1d(np.insert(self.q[self.minima],0,0), 
                                         np.insert(smoothed_y[self.minima],0,0),
                                         kind='linear', bounds_error=False, fill_value='extrapolate')
        elif self.baseline_type == "spline" and len(self.minima) >= 5:
            baseline_func = make_smoothing_spline(np.insert(self.q[self.minima],0,0), 
                                                  np.insert(smoothed_y[self.minima],0,0))
            
        else:
            baseline_func = lambda x: np.zeros_like(x)
        
        
        # Evaluate the baseline function over the entire q range
        if self.baseline_type == "pchip":
            baseline_values = pchip_interpolate(np.insert(self.q[self.minima],0,0), 
                                            np.insert(smoothed_y[self.minima],0,0), 
                                            self.q)
        else:
            baseline_values = baseline_func(self.q)
        
        # Set negative baseline values to zero
        baseline_values = np.maximum(baseline_values, 0)
        # Take the minimum of baseline values and the data
        baseline_values = np.minimum(baseline_values, smoothed_y)
        self.baseline=baseline_values
        #subtraction of baseline
        y_bl = np.abs(y - self.baseline)
        self.Rq4_bl=y_bl
        smoothed_y_bl = np.abs(smoothed_y - self.baseline)
        self.smoothed_Rq4_bl=smoothed_y_bl
        sqrt_y_bl=np.sqrt(np.copy(y_bl))
        sqrt_smoothed_y_bl=np.sqrt(np.copy(smoothed_y_bl))

        # sign flipping after each minima
        for idx in self.minima:
            sqrt_y[idx:] *= -1
            sqrt_y_bl[idx:] *= -1
            sqrt_smoothed_y_bl[idx:] *= -1
        return sqrt_y,sqrt_y_bl,sqrt_smoothed_y_bl
        
    def set_no_background(self):
        self.tth_bg = self.tth
        self.det_bg = np.zeros_like(self.det)
        self.mon_bg = np.ones_like(self.det)
        self.integrationTime_bg = np.zeros_like(self.det)

        
    def get_rawdata(self):
        # read the signal XRR
        if self.datafilePathXRR.endswith(".h5"):
            self.tth, self.det, self.mon, self.integrationTime, self.start_time_str, self.end_time_str = self.read_h5(self.datafilePathXRR, self.scanNbrXRR)
            if self.scanNbrBG != None:  
                self.tth_bg, self.det_bg, self.mon_bg, self.integrationTime_bg, self.start_time_str_bg, self.end_time_str_bg = self.read_h5(self.datafilePathBG,self.scanNbrBG)
                if self.BG_subrange != None and len(self.BG_subrange) == 2:
                    # select a subset of the background (e.g. the last few points) 
                    m, M = self.BG_subrange[0],self.BG_subrange[1]
                    self.tth_bg_sub = self.tth_bg[m:M]
                    self.det_bg_sub = self.det_bg[m:M]
                    self.mon_bg_sub = self.mon_bg[m:M]
                    self.integrationTime_bg_sub = self.integrationTime_bg[m:M]
                    # and reinterpolate over the whole range
                    x = self.tth
                    x_bg_sub = self.tth_bg_sub
                    self.tth_bg = interp1d(x_bg_sub, self.tth_bg_sub, kind='linear', bounds_error=False, fill_value='extrapolate')(x)
                    self.det_bg = interp1d(x_bg_sub, self.det_bg_sub, kind='linear', bounds_error=False, fill_value=(self.det_bg_sub[0],self.det_bg_sub[-1]))(x)
                    self.mon_bg = interp1d(x_bg_sub, self.mon_bg_sub, kind='linear', bounds_error=False, fill_value=(self.mon_bg_sub[0],self.mon_bg_sub[-1]))(x)
                    self.integrationTime_bg = interp1d(x_bg_sub, self.integrationTime_bg_sub, kind='linear', bounds_error=False, fill_value='extrapolate')(x)

            else:
                self.set_no_background()
        
        else :
            sf = sr.SpecFile(self.datafilePathXRR)
            s = sr.Scan(sf,self.scanNbrXRR)
            self.tth, self.det, self.mon, self.integrationTime = s.Psi, s.detc, s.Monitor4, s.Seconds
            if self.scanNbrBG == None:
                self.set_no_background()
            else:
                if self.scanNbrBG == "auto":
                    self.scanNbrBG = self.scanNbrXRR+1
                else:
                    pass    
                s_bg = sr.Scan(sf,self.scanNbrBG)
                self.tth_bg, self.det_bg, self.mon_bg, self.integrationTime_bg = s_bg.Psi, s_bg.detc, s_bg.Monitor4, s_bg.Seconds    
                
    
    def preprocess(self):
        self.mondc = self.mon-self.integrationTime*self.darkCurrent
        self.detm = self.det/self.mondc
        if self.directBeamAmplitude == None :
            self.fit_directBeam()
            
        self.tthc = self.tth-self.tthOffset
        self.detmn = self.detm/self.directBeamAmplitude

        self.fpc = self.footPrintCorrection(θ_in = self.tthc/2)
        self.detmnp = self.detmn/self.fpc

        # restrict to away from direct beam
        self.imin,self.imax = np.where(self.tthc>self.tthmin)[0].min(),-1
        self.tthcr, self.detmnpr = self.tthc[self.imin:self.imax], self.detmnp[self.imin:self.imax]

        
        self.mondc_bg = self.mon_bg-self.integrationTime_bg*self.darkCurrent
        self.detm_bg = self.det_bg/self.mondc_bg
        self.tthc_bg = self.tth_bg-self.tthOffset
        self.detmn_bg = self.detm_bg/self.directBeamAmplitude
        self.fpc_bg = self.footPrintCorrection(θ_in = self.tthc_bg/2)
        self.detmnp_bg = self.detmn_bg/self.fpc_bg
        self.imin_bg,self.imax_bg = np.where(self.tthc_bg>self.tthmin_bg)[0].min(),-1
        self.tthcr_bg, self.detmnpr_bg = self.tthc_bg[self.imin_bg:self.imax_bg], self.detmnp_bg[self.imin_bg:self.imax_bg]
        self.detmnpri_bg = interp1d(self.tthcr_bg, self.detmnpr_bg, kind='linear', bounds_error=False,  fill_value=(self.detmnpr_bg[:10].mean(),self.detmnpr_bg[-10].mean()))(self.tthcr)

        self.qraw = self.tth2q(self.tthcr, wavelength=self.wavelength)
        self.Rraw = self.detmnpr - self.detmnpri_bg



    def fit_directBeam(self):
        self.directBeamAmplitude = self.detm.max()
        
    
    def fit_directBeam_dev(self,x,y):
        # using the erf rectangle model (= convolution of gaussian with slits, typical width is vg5/800 mm)
        self.model = RectangleModel(form='erf')
        # guess the parameters
        amplitude = y.max()
        center = x[y.argmax()]
        yp = np.gradient(y) # derivate to find the inflexion points
        center1,center2 = x[yp.argmax()],x[yp.argmin()]
        sigma1,sigma2=self.directBeamFHWM, self.directBeamFHWM
        self.params = self.model.make_params(amplitude=amplitude, center1=center1, center2=center2, sigma1=sigma1, sigma2=sigma2)
        # fix the FWHM
        self.params["sigma1"].set(vary=False)
        self.params["sigma2"].set(vary=False)
        # do the fit
        self.result = self.model.fit(y, params, x=x)
        self.directBeamAmplitude = self.result.best_values['amplitude']
        self.sigma1,self.sigma2 = self.result.best_values['sigma1'],self.result.best_values['sigma1']
        self.tthOffset = (result.best_values['center1']+result.best_values['center2'])/2



    def footPrintCorrection(self, θ_in):
        # returns the factor to correct for the illuminated area
        # l = sample length (mm)
        # b_in = beam width (mm)
        # θ_in = array of incidence angles (degree)
        self.θo_in = deg(np.arcsin(self.beamSize/self.sampleLength))
        fpc = abs(np.sin(rad(θ_in))/(self.beamSize/self.sampleLength)*(θ_in<self.θo_in))+\
                   1.*(θ_in>=self.θo_in)
        return fpc

    

    def ensure_positivity(self):
        self.qc = self.qraw[np.where(self.Rraw>0)[0]]
        self.Rc = self.Rraw[np.where(self.Rraw>0)[0]]
    
    def crop(self):
        if self.qmax == None:
            self.q = self.qc.copy()
            self.R = self.Rc.copy() 
        else:
            self.q = self.qc[np.where(self.qc<self.qmax)[0]].copy()
            self.R = self.Rc[np.where(self.qc<self.qmax)[0]].copy()    

    def read_h5(self, h5filePath, scanNbr):
        self.ensure_broadcast(self.datafilePathXRR, scanNbr)
        with silx.io.h5py_utils.File(h5filePath, 'r') as f:
            #tth = f[f"{scanNbr}.1/measurement/psi"][()]
            #det = f[f"{scanNbr}.1/measurement/detcor"][()]
            #mon = f[f"{scanNbr}.1/measurement/mon4"][()]
            #integrationTime = f[f"{scanNbr}.1/measurement/integration_time"][()]
            tth = self.dict_counter['psi']
            mon = self.dict_counter['mon4']
            attn = self.dict_counter['attn'].astype(int)
            
            # check the attenuation coefficients
            self.energy = f[f'{scanNbr}.1/instrument/positioners/energy'][()]
            
            if abs(self.energy-27)>0.010:
                raise ValueError("Energies different than 27 keV are not supported yet")
            det = self.dict_counter['det']*self.attnFactors27keV[attn]
            start_time_str = f[f"{scanNbr}.1/start_time"][()]
            #self.start_time = datetime.datetime.fromisoformat(self.start_time_str.decode()) 
            end_time_str = f[f"{scanNbr}.1/end_time"][()]
            #self.end_time = datetime.datetime.fromisoformat(self.end_time_str.decode()) 
            try:
                integrationTime = self.dict_counter['integration_time']
            except KeyError:
                s=f[f"{scanNbr}.1"]
                scan_title = str(s['title'][()])
                #print(scan_title.split())
                try:
                    integrationTime = float(scan_title.split()[-1])*np.ones_like(tth)
                except ValueError:
                    integrationTime = float(scan_title.split()[-1][:-1])*np.ones_like(tth)
        return tth, det, mon, integrationTime, start_time_str, end_time_str

    def ensure_broadcast(self,h5filePath, scanNbr):
        #TODO : refactor such that reading and 0-padding are separated, and make 0-pad more general
        self.dict_counter= {} 
        with silx.io.h5py_utils.File(h5filePath, 'r') as f:
            counters = list(f[f"{scanNbr}.1/measurement"].keys())
            counters_len = [len(f[f"{scanNbr}.1/measurement/{counter}"][()]) for counter in counters]
            counters_len.sort()
            for counter in counters:
                if len(f[f"{scanNbr}.1/measurement/{counter}"][()]) != counters_len[-1]:
                    self.dict_counter[counter] = np.append(f[f"{scanNbr}.1/measurement/{counter}"][()],0)
                else: 
                    self.dict_counter[counter] = f[f"{scanNbr}.1/measurement/{counter}"][()]


      

    def remove_qvalues(self):
        for qvaluesTuple in self.qvaluesRangesToRemove:
            # indices out of the range to remove (= indices to keep)
            # i.e. q < qmin or q > qmax with [qmin,qmax] the tuple in the qvaluesRangesToRemove
            idx_keep = np.where((self.q<qvaluesTuple[0])+(self.q>qvaluesTuple[1]))[0]
            self.q2 = self.q[idx_keep]
            self.R2 = self.R[idx_keep]
            
            # quadratic interpolation
            # for the interpolation, keep only a few points near the range (but still outside):
            # i.e. q > qmin-.02 and q < qmax+.02
            idx_i = np.where((self.q2>qvaluesTuple[0]-.02)*(self.q2<qvaluesTuple[1]+0.02))[0]
            x = self.q2[idx_i]
            y = x**4*self.R2[idx_i]
            m = QuadraticModel()
            r = m.fit(y, x=x)
            
            # copy the original array 
            self.Ri = self.R.copy()
            # indices inside the range (= indices to replace)
            # i.e. q > qmin and q < qmax
            idx_replace = np.where((self.q>qvaluesTuple[0])*(self.q<qvaluesTuple[1]))[0]
            self.R[idx_replace] = r.eval(x=self.q[idx_replace])/self.q[idx_replace]**4


    def tth2q(self, tth, wavelength = 12398/27000):
        return 4*np.pi*np.sin(rad(tth/2))/wavelength

    

    def export(self, x="q", y="R", filename=None):
        """
        choose which x,y values to export:
        x :
           "tth" : raw two theta angles
           "tthc" : corrected two theta angles
           "tthcr" : corrected two theta angles and restricted to away from direct beam
           "qraw" : q values (from tthcr)
           "qc" :  q values from qraw for only positive R positions
           "q" : q values from qc, cropped to a maximum qmax value (if needed)

        y :
           "det" : raw detector signal
           "detm" : det normalised to the monitor
           "detmn" : detm normalised to the directbeam
           "detmnp" : detmn corrected for the footprint
           "detmnpr" : detmnp restricted to away from direct beam
           "Rraw" : detmnpr corrected from the background
           "Rc" : Rraw limited to positive values
           "R" : reflection coeff from Rc, cropped to a maximum qmax value (if needed)
        """
        if len(getattr(self,x)) != len(getattr(self,y)):
            print(f"{x} and {y} do not have the same length ! aborting")
            raise ValueError 
        if filename == None:
            filename, file_extension = os.path.splitext(self.datafilePathXRR)
            filename = filename+".dat"
            #print(f"exporting to {filename}")
        with open(filename,"w") as f:
            f.write(f"#{x} {y}\n")
            for iii in np.arange(len(getattr(self,x))):
                f.write(f"{getattr(self,x)[iii]} {getattr(self,y)[iii]}\n")

    def export2(self, x_list=["q"], y_list=["R"], filename=None):
        """
        choose which x,y values to export:
                x :
           "tth" : raw two theta angles
           "tthc" : corrected two theta angles
           "tthcr" : corrected two theta angles and restricted to away from direct beam
           "qraw" : q values (from tthcr)
           "qc" :  q values from qraw for only positive R positions
           "q" : q values from qc, cropped to a maximum qmax value (if needed)
           "z" : z values from ift
        x_list: list of y values to export (default is ["q"])

        y :
           "det" : raw detector signal
           "detm" : det normalised to the monitor
           "detmn" : detm normalised to the directbeam
           "detmnp" : detmn corrected for the footprint
           "detmnpr" : detmnp restricted to away from direct beam
           "Rraw" : detmnpr corrected from the background
           "Rc" : Rraw limited to positive values
           "R" : reflection coeff from Rc, cropped to a maximum qmax value (if needed)
           "deltaRho_nc": deltaRho without any processing
           "deltaRho": deltaRho with signal inverse
           "deltaRho_bl_s": deltaRho with signal inverse, bl subtration and smooth process 

        y_list: list of y values to export (default is ["R"])
        """
        for x,y in zip(x_list,y_list):
            if len(getattr(self, x)) != len(getattr(self, y)):
                print(f"{x} and {y} do not have the same length! Aborting.")
                raise ValueError

        if filename is None:
            filename, file_extension = os.path.splitext(self.datafilePathXRR)
            filename = filename + ".dat"

        with open(filename, "w") as f:
            f.write(f"#start time : {self.start_time_str}, end time : {self.end_time_str}\n")
            f.write(f"#{' '.join(x_list)} {' '.join(y_list)}\n")

            for iii in np.arange(len(getattr(self, x_list[0]))):
                x_values = [str(getattr(self, x)[iii]) for x in x_list]
                y_values = [str(getattr(self, y)[iii]) for y in y_list]
                f.write(f"{' '.join(x_values)} {' '.join(y_values)}\n")

    def export_to_hdf5(self,h5_filename=None):
        """
        Export selected attributes to an HDF5 file.
        """
        if h5_filename is None:
            h5_filename, file_extension = os.path.splitext(self.datafilePathXRR)
            h5_filename = h5_filename + ".h5"

        with h5.File(h5_filename, "w") as h5_file:
            # Store sample name and XRR scanNbr as attributes of the root group
            h5_file.attrs["sampleName"] = self.sampleName
            h5_file.attrs["scanNbrXRR"] = self.scanNbrXRR

            # Store numerical attributes (np.arrays) as datasets: 
            # tth,tthc,qraw,qc,q,z,det,detm,detmn,detmnp,detmnpr,Rraw,Rc,R,deltaRho_nc,deltaRho,deltaRho_bl,deltaRho_bl_s
            # create dataset group XRR
            dataset_XRR = h5_file.create_group("dataset_XRR")
            # Create "x counters" and "y counters" subdatasets within dataset_XRR, and save the related datasets
            x_counters_XRR = dataset_XRR.create_group("x counters")
            y_counters_XRR = dataset_XRR.create_group("y counters")
            x_counters_XRR.create_dataset("tthcr", data=self.tthcr)
            x_counters_XRR.create_dataset("qraw", data=self.qraw)
            y_counters_XRR.create_dataset("self.detmnpr", data=self.detmnpr)
            y_counters_XRR.create_dataset("self.Rraw", data=self.Rraw)

            # create dataset group EDP
            dataset_EDP = h5_file.create_group("dataset_EDP")
            # Create "x counters" and "y counters" subdatasets within dataset_XRR, and save the related datasets
            x_counters_EDP = dataset_EDP.create_group("x counters")
            y_counters_EDP = dataset_EDP.create_group("y counters")
            x_counters_EDP.create_dataset("z", data=self.z)
            y_counters_EDP.create_dataset("deltaRho_nc", data=self.deltaRho_nc)
            y_counters_EDP.create_dataset("deltaRho", data=self.deltaRho)
            y_counters_EDP.create_dataset("deltaRho_bl", data=self.deltaRho_bl)
            y_counters_EDP.create_dataset("deltaRho_bl_s", data=self.deltaRho_bl_s)





        print(f"Selected data exported to {h5_filename}")

    def FourierBasicInverse(self):
        for j in range(0, len(self.yout)):
            self.yout[j] = 0
            zj = -self.zrange//2 + j / (len(self.yout)) * self.zrange
            self.xout[j] = zj
            imin = 0
            imax = len(self.yout) - 1
            for i in range(imin, imax):
                if self.xin[i] > 0:
                    self.yout[j] += self.yin[i] * 2 * np.sin(self.xin[i] * zj) * (self.xin[i + 1] - self.xin[i])
                i += 1
            j += 1
        self.yout /= (2 * np.pi)


    
    def ift(self,q,sqrtRq4):
        re = 2.8179403262E-5 #Angstrom
        prefactor = 4*np.pi*re
        self.xin = q
        self.yin = sqrtRq4/prefactor
        self.xout = np.zeros(len(self.xin))
        self.yout = np.zeros(len(self.xin))
        self.FourierBasicInverse() #self.yout = gradient_rho
      

        self.yout_INT = np.zeros(len(self.yout))
        self.yout_INT[0]=0
        for k in range(1, len(self.yout)-1):
            M=[self.yout[k], self.yout[k+1]]
            N=[self.xout[k], self.xout[k+1]]

            self.yout_INT[k] = self.yout_INT[k-1] + np.trapz(M, x=N)#, dx=0.35087719)
            k +=1

        self.yout_d = self.yout_INT  
        self.avg = np.mean(self.yout_d)

        if (self.yout_d[len(self.yout_d) // 2] > 0):
            self.yout_d *= -1

        self.result_int = np.vstack((self.xout, self.yout_d)).T

        #self.z, self.deltaRho = self.xout, self.yout_d
        return self.xout, self.yout_d

    def IFT_slow(self,sqrt_Rq4,q,zrange):
        """
        Compute the discrete inverse Fourier Transform of the 1D array sqrt_Rq4
        sqrt_Rq4,q are 1D arrays and zrange is integer
        return d_rho and z, which are both 1D arrays
        """
        zlist =  np.linspace(-zrange / 2, zrange / 2, len(q)) #define a zlist centered at zero
        re = 2.8179403262E-5 #Angstrom
        prefactor = 4*np.pi*re
        dq = np.abs(q.max()-q.min())/len(q)
        M2=np.array([np.exp(-1.j * q * z ) for z in zlist])
        IFT = np.multiply(M2,sqrt_Rq4/prefactor)
        d_rho = -np.trapz(IFT,dx=dq)/(np.pi)
        
        return d_rho,zlist

    def plot_analysis(self, figsize = (6,2)):
        #figure 1 
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.tth, self.det, label=("XRR"))
        ax.plot(self.tth_bg, self.det_bg, label=("BG"))
        ax.set_yscale('log')
        plt.legend()
        ax.set_xlabel("tth (deg)")
        ax.set_ylabel("det (arb. unit)")
        ax.set_title("fig.1: raw data")
        plt.show()
        #figure 2 
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.tth, self.detm, label=("XRR"))
        ax.plot(self.tth_bg, self.detm_bg, label=("BG"))
        ax.set_yscale('log')
        plt.legend()
        ax.set_xlabel("tth (deg)")
        ax.set_ylabel("det/mon (arb. unit)")
        ax.set_title("fig.2: normalized to monitor")
        plt.show()
        #figure 3
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.tthc, self.detmn, label=("XRR"))
        ax.plot(self.tthc_bg, self.detmn_bg, label=("BG"))
        ax.set_yscale('log')
        plt.legend()
        ax.set_xlabel("tth (deg)")
        ax.set_ylabel("det/mon/db (arb. unit)")
        ax.set_title("fig.3: normalized to direct beam")
        plt.show()
        #figure 4 
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.tthc, self.detmn, label=("XRR-no fc"))
        ax.plot(self.tthc, self.detmnp, label=("XRR"))
        ax.plot(self.tthc_bg, self.detmnp_bg, label=("BG"))
        ax.set_yscale('log')
        ax.set_xlabel("tth (deg)")
        ax.set_ylabel("det/mon/db/fpc (arb. unit)")
        ax.plot(self.tthcr, self.detmnpr, label=("XRR (out of db)"))
        ax.plot(self.tthcr_bg, self.detmnpr_bg, label=("BG (out of db)"))
        ax.plot(self.tthcr, self.detmnpri_bg, label=("BG (out of db) interpolated"))
        plt.legend()
        ax2 = ax.twinx()
        ax2.plot(self.tthc,self.fpc,'k:', label="footprint correction")
        ax2.set_ylabel("footprint correction")
        ax.set_title("fig.4: normalized to db + footprint correction + bg interp")
        plt.show()

        #figure 5
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.q, self.R, label=("R"))
        ax.plot(self.qc, self.Rc, label=("R (R>0)"))
        ax.set_yscale('log')
        plt.legend()
        ax.set_xlabel("q (A-1)")
        ax.set_ylabel("R ")
        ax.set_title("fig.5: R(q)")
        plt.show()

        #figure 6
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.qraw, self.Rraw*(self.qraw**4), label=("q4R (raw)"))
        ax.plot(self.qc, self.Rc*(self.qc**4), label=("q4R (R>0) (raw)"))
        plt.legend()
        ax.set_xlabel("q (A-1)")
        ax.set_ylabel("q4R ")
        ax.set_title("fig.6: q4R(q) (raw)")
        ax.hlines(0, *ax.get_xlim(),colors="k")
        ax.set_xlim( *ax.get_xlim())
        plt.show()

        #figure 7
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.qc, self.Rc*(self.qc**4),'k', label=("q4R (R>0) (raw)"))
        ax.plot(self.q,self.Rq4_bl,'b',label=("q4R_bl"))
        ax.plot(self.q,self.smoothed_Rq4_bl,'r',label=("q4R_bl_s"))
        ax.plot(self.q[self.minima],self.smoothed_Rq4[self.minima],'gD', markersize=4,label=("minima"))
        ax.plot(self.q,self.baseline,'ko-', markersize=1,lw=1,label=("baseline"))
        plt.legend()
        ax.set_xlabel("q (A-1)")
        ax.set_ylabel("q4R ")
        ax.set_title("fig.7: q4R(q)")
        ax.hlines(0, *ax.get_xlim(),colors="k")
        ax.set_xlim( *ax.get_xlim())
        plt.show()
        
        #figure 8
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.q, np.sqrt(self.R*self.q**4), 'k',label=("sqrt q4R_nc"))
        ax.plot(self.q,self.sqrtRq4,'b',label=("sqrt q4R"))
        ax.plot(self.q,self.sqrt_smoothed_Rq4_bl,'r',label=("sqrt q4R_bl_s"))
        plt.legend()
        ax.set_xlabel("q (A-1)")
        ax.set_ylabel("sqrt_q4R")
        ax.set_title("fig.8: sqrt q4R(q)")
        ax.hlines(0, *ax.get_xlim(),colors="k")
        ax.set_xlim( *ax.get_xlim())
        plt.show()
        
        #figure 9
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.z_nc, self.deltaRho_nc,'k*-',ms=1,label=("EDP_nc"))
        ax.plot(self.z, self.deltaRho,'bo-',ms=1, label=("EDP_inverted"))
        ax.plot(self.z_bl, self.deltaRho_bl_s,'ro-',ms=1, label=("EDP_inverted_bl_s"))
        plt.legend()
        ax.set_xlabel("z (A)")
        ax.set_ylabel("deltaRho (e-/A3)")
        ax.set_title("fig.9: Electron density profile")
        ax.grid()
        plt.show()


    def save_plot_analysis(self, filename, figsize = (11,8), dpi = 150,
                           tthmin = -.25, tthmax = 4,
                           Irawmin = 1, Irawmax = 2E10,
                           Imonmin = 1E-5, Imonmax = 2E5,
                           Inormmin = 1E-10, Inormmax = 2,
                           qmin = 0, qmax = 0.9, 
                           Rmin = 1E-10, Rmax = 1,
                           q4Rmin = -1E-9, q4Rmax = 6E-8, 
                           sqrtq4Rmin = -0.00025, sqrtq4Rmax = 0.00025, 
                           rhomin = -.7, rhomax = 0.05):
        
        zmin = -self.zrange/2
        zmax = self.zrange/2

        plt.ioff()
        fig,axs = plt.subplots(3, 3, figsize=figsize)
        #figure 1 
        ax=axs[0,0]
        ax.plot(self.tth, self.det, label=("XRR"))
        ax.plot(self.tth_bg, self.det_bg, label=("BG"))
        ax.set_yscale('log')
        ax.set_xlim(tthmin,tthmax)
        ax.set_ylim(Irawmin,Irawmax)
        ax.set_xlabel("tth (deg)", fontsize="small")
        ax.set_ylabel("det (arb. unit)", fontsize="small")
        ax.set_title("fig.1: raw data", fontsize="small")
        
        #figure 2 
        ax=axs[1,0]
        ax.plot(self.tth, self.detm, label=("XRR"))
        ax.plot(self.tth_bg, self.detm_bg, label=("BG"))
        ax.set_yscale('log')
        ax.set_xlim(tthmin,tthmax)
        ax.set_ylim(Imonmin,Imonmax)
        ax.set_xlabel("tth (deg)", fontsize="small")
        ax.set_ylabel("det/mon (arb. unit)", fontsize="small")
        ax.set_title("fig.2: normalized to monitor", fontsize="small")
        
        #figure 3
        ax=axs[2,0]
        ax.plot(self.tthc, self.detmn, label=("XRR"))
        ax.plot(self.tthc_bg, self.detmn_bg, label=("BG"))
        ax.set_yscale('log')
        ax.set_xlim(tthmin,tthmax)
        ax.set_ylim(Inormmin,Inormmax)
        ax.set_xlabel("tth (deg)", fontsize="small")
        ax.set_ylabel("det/mon/db (arb. unit)", fontsize="small")
        ax.set_title("fig.3: normalized to direct beam", fontsize="small")
        
        #figure 4 
        ax=axs[0,1]
        ax.plot(self.tthc, self.detmn, label=("XRR-no fc"))
        ax.plot(self.tthc, self.detmnp, label=("XRR"))
        ax.plot(self.tthc_bg, self.detmnp_bg, label=("BG"))
        ax.set_yscale('log')
        ax.set_xlim(tthmin,tthmax)
        ax.set_ylim(Inormmin,Inormmax)
        ax.set_xlabel("tth (deg)", fontsize="small")
        ax.set_ylabel("det/mon/db/fpc (arb. unit)", fontsize="small")
        ax.plot(self.tthcr, self.detmnpr, label=("XRR (out of db)"))
        ax.plot(self.tthcr_bg, self.detmnpr_bg, label=("BG (out of db)"))
        ax.plot(self.tthcr, self.detmnpri_bg, label=("BG (out of db) interp."))
        ax2 = ax.twinx()
        ax2.plot(self.tthc,self.fpc,'k:', label="footprint correction")
        ax2.set_ylabel("footprint correction", fontsize="small")
        ax.set_title("fig.4: normalized to db + footprint correction + bg interp", fontsize="small")

        #figure 5
        ax=axs[1,1]
        ax.plot(self.q, self.R, label=("R"))
        ax.plot(self.qc, self.Rc, label=("R>0"))
        ax.set_xlim(qmin,qmax)
        ax.set_ylim(Rmin,Rmax)
        ax.set_yscale('log')
        ax.set_xlabel("q (A-1)", fontsize="small")
        ax.set_ylabel("R ", fontsize="small")
        ax.set_title("fig.5: R(q)", fontsize="small")

        #figure 6
        ax=axs[2,1]
        ax.plot(self.qraw, self.Rraw*(self.qraw**4), label=("raw"))
        ax.plot(self.qc, self.Rc*(self.qc**4), label=("R>0"))
        ax.set_xlim(qmin,qmax)
        ax.set_ylim(q4Rmin,q4Rmax)
        ax.set_xlabel("q (A-1)", fontsize="small")
        ax.set_ylabel("q4R ", fontsize="small")
        ax.set_title("fig.6: q4R(q) (raw)", fontsize="small")

        #figure 7
        ax=axs[0,2]
        ax.plot(self.qc, self.Rc*(self.qc**4),'k')
#        ax.plot(self.qc, self.Rc*(self.qc**4),'k', label=("q4R (R>0)"))
        #ax.plot(self.q,self.Rq4_bl,'b',label=("bl"))
        #ax.plot(self.q,self.smoothed_Rq4_bl,'r',label=("bl+s"))
        #ax.plot(self.q[self.minima],self.smoothed_Rq4[self.minima],'go', markersize=4,label=("minima"))
        #ax.plot(self.q,self.baseline,'go-', markersize=1,lw=1,label=("baseline"))
        ax.set_xlim(qmin,qmax)
        ax.set_ylim(q4Rmin,q4Rmax)
        ax.set_xlabel("q (A-1)", fontsize="small")
        ax.set_ylabel("q4R ", fontsize="small")
#        ax.set_title("fig.7: q4R(q)", fontsize="small")

        #figure 8
        ax=axs[1,2]
        #ax.plot(self.q, np.sqrt(self.R*self.q**4), 'k',label=("nc"))
        #ax.plot(self.q,self.sqrtRq4,'b',label=("ph."))
        #ax.plot(self.q,self.sqrt_smoothed_Rq4_bl,'r',label=("ph.+bl"))
        ax.plot(self.q,self.sqrt_smoothed_Rq4_bl,'k')
        ax.set_xlim(qmin,qmax)
        ax.set_ylim(sqrtq4Rmin,sqrtq4Rmax)
        ax.set_xlabel("q (A-1)", fontsize="small")
        ax.set_ylabel(r"$\sqrt{q4R}$", fontsize="small")
#        ax.set_title("fig.8: sqrt q4R(q)", fontsize="small")

        #figure 9
        ax=axs[2,2]
#        ax.plot(self.z_nc, self.deltaRho_nc,'k*-',ms=1,label=("nc"))
#        ax.plot(self.z, self.deltaRho,'bo-',ms=1, label=("ph."))
#        ax.plot(self.z_bl, self.deltaRho_bl_s,'ro-',ms=1, label=("ph. + bl"))
        ax.plot(self.z_bl, self.deltaRho_bl_s,'k',ms=1)
        ax.set_xlim(zmin,zmax)
        ax.set_ylim(rhomin,rhomax)
        ax.set_xlabel("z (A)", fontsize="small")
        ax.set_ylabel("deltaRho (e-/A3)", fontsize="small")
#        ax.set_title("fig.9: Electron density profile", fontsize="small")

        for ax in axs.flatten():
          #  ax.legend(fontsize="small")
            ax.grid()

        fig.suptitle(self.sampleName)
        fig.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close(fig)

            


def parse_scantitle(scan_title):
    scan_args = scan_title.split(' ')
    scan_elements = {}
    scan_elements["type"] = str(scan_args[0])
    if scan_elements["type"] == "a2scan":
        motor1, motor1_start, motor1_end = scan_args[1], scan_args[2], scan_args[3]
        scan_elements[f"{motor1}_start"] = float(motor1_start) 
        scan_elements[f"{motor1}_end"] = float(motor1_end) 
        motor2, motor2_start, motor2_end = scan_args[4], scan_args[5], scan_args[6]
        scan_elements[f"{motor2}_start"] = float(motor2_start) 
        scan_elements[f"{motor2}_end"] = float(motor2_end)         
        scan_elements["points"] = scan_args[7]
        scan_elements["counting time"] = scan_args[8]
    else:
        pass
    return scan_elements
        

def isXRR_from_scantitle(scan_title):
    scan_elements = parse_scantitle(scan_title)
    if scan_elements["type"] == "a2scan":
        return scan_elements["bg1_start"]*2 == scan_elements["psi_start"] and scan_elements["bg1_end"]*2 == scan_elements["psi_end"]
    
def isBG_from_scantitle(scan_title):
    scan_elements = parse_scantitle(scan_title)
    if scan_elements["type"] == "a2scan":
        return scan_elements["bg1_start"]*2 != scan_elements["psi_start"] and scan_elements["bg1_end"]*2 != scan_elements["psi_end"]
    
    
def build_dictScan_from_master_h5(h5FileExp, filters = ["measurement",], verbose=False):
    list_dictScan=[]
    item = 0
    with silx.io.h5py_utils.File(h5FileExp,'r') as f:
        keys = f.keys()
        for k in keys:
            do_process = np.array([(keyword in k) for keyword in filters]).all()
            if do_process:
                if verbose: print("process ", k)
                try:
                    s=f[k]
                    scan_title = s['title'][()].decode()
                    if verbose : print(scan_title)
                    #check if it is a XRR scan
                    if isXRR_from_scantitle(scan_title):
                        prefix = "_".join(k.split("_")[:-1])
                        scan_num = k.split("_")[-1].split(".")[0]
                        if verbose : print("found xrr", prefix, scan_num)
                        dictScan={}
                        list_dictScan.append(dictScan)
                        dictScan['item'] = item
                        item=item+1
                        dictScan['h5filePath']=f[k].file.filename
                        dictScan['sampleName']=os.path.split(os.path.split(os.path.split(dictScan['h5filePath'])[0])[0])[1]
                        dictScan['title']=s['title'][()]
                        dictScan['scanNbrXRR']=int(scan_num.split('.')[0])

                        #check if it has a following background (BG) scan
                        #if the type of f[f'{scan_numBG}.1']['title'][()] is bytes
                        dictScan['scanNbrBG']= None
                        scanBG_num=int(scan_num)+1
                        k_bg = f"{prefix}_{scanBG_num}.1"
                        if k_bg in keys:
                            if verbose: print("bg name ", k_bg)
                            scanBG_title = f[k_bg]['title'][()].decode()
                            if verbose: print("scanBG_title ", scanBG_title)
                            if isBG_from_scantitle(scanBG_title):
                                if verbose : print("found xrr BG")
                                dictScan['scanNbrBG']=scanBG_num
                                dictScan['title_BG']=scanBG_title
                except Exception as e:
                    print(e)
                    pass
                        
    return list_dictScan



            
def dict_scan(h5File_list, verbose = False):
    list_dictScan=[]
    item = 0
    for h5filePath in h5File_list:
        with silx.io.h5py_utils.File(h5filePath,'r') as f:
            listScans = list(f.keys())
            listScans.sort(key= lambda x: int(x.split(".")[0])) #sort the listScan with scan numbers
            for scan_num in listScans:
                dictScan={}
                s=f[scan_num]
                scan_title = s['title'][()].decode()
                if verbose : print(scan_title)
                #the type of s['title'][()] is bytes
                
                #check if it is a XRR scan
                if isXRR_from_scantitle(scan_title):
                    if verbose : print("found xrr", scan_num)
                    list_dictScan.append(dictScan)
                    dictScan['item'] = item
                    item=item+1
                    dictScan['h5filePath']=h5filePath
                    dictScan['sampleName']=os.path.split(h5filePath)[1][:-3]
                    dictScan['title']=s['title'][()]
                    dictScan['scanNbrXRR']=int(scan_num.split('.')[0])

                    #check if it has a following background (BG) scan
                    #if the type of f[f'{scan_numBG}.1']['title'][()] is bytes
                    scanBG_num=int(scan_num.split('.')[0])+1
                    scanBG_title = f[f'{scanBG_num}.1']['title'][()].decode()
                    if isBG_from_scantitle(scanBG_title):
                        if verbose : print("found xrr BG")
                        dictScan['scanNbrBG']=scanBG_num
                        dictScan['title_BG']=f[f'{scanBG_num}.1']['title'][()]
                    else:
                        dictScan['scanNbrBG']= None
    return list_dictScan


def DFT_slow(d_rho,z,qmin,qmax):
    """
    Compute the discrete Fourier Transform of the 1D array d_rho
    d_rho,z are 1D array and qmin, qmax are float
    return sqrt_Rq4 and q, which are both 1D array
    """
    qlist =np.arange(qmin,qmax,np.abs(qmax-qmin)/(len(z)))
    re = 2.8179403262E-5 #Angstrom
    prefactor2 = 4*np.pi*re
    dz = z.max()*2/len(z)
    M = np.array([np.exp(1.j * q * z ) for q in qlist])
    FT = np.multiply(M,d_rho)*prefactor2
    sqrt_Rq4 = np.trapz(FT,dx=dz)
    
    return sqrt_Rq4, qlist

def just_invert(peaks,minima,y):
    ###In the case that invert_bl doesn't work, define peaks and minima first and use this function to invert
    #find the index of the peak with high amplitude
    max_peak_idx = np.argmax(y[peaks[:len(peaks)//2+1]]) 
    # For the minima on the left side of the max peak
    invert = True
    for i in range(max_peak_idx-1,-1,-1):
        if i != 0:
            start_idx = minima[i-1]
        else:
            start_idx = minima[i]
        end_idx = minima[i]
        if invert:
            y[start_idx:end_idx] *= -1
        invert = not invert 
    # For the minima on the right side of the max peak
    invert2 = True
    for i in range(max_peak_idx,len(minima)-1):
        start_idx = minima[i]
        end_idx = minima[i+1]
        if invert2:
            y[start_idx:end_idx] *= -1
        invert2 = not invert2 
    return y