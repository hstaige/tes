import math
import copy
import re
import numpy as np
import pyqtgraph as pg
from scipy.signal import find_peaks
from lmfit import Parameters, minimize, fit_report
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# 'data' should be an np array with columns of energy, time
dir = '/home/tim/research/EBIT-TES-Data/data_by_state/'
run = '20231015_0000'
states = ['G','H','I']
data = np.empty((3,0))
for state in states:
    data = np.hstack((data,np.load(f'{dir}{run}_{state}.npy')))
data = data[(0,2),:].T

##### defaults: #####
default_binsize = 0.25
default_xrange = [600, 1850]
default_peak_prom = 500
default_sigma = 1.95
default_gamma = 5.37

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def vect_voigt(x,A,mu,sigma,gamma):
    x_len = len(x)
    param_len = len(A)
    A_arr = np.tile(A,(x_len,1))
    mu_arr = np.tile(mu,(x_len,1))
    sigma_arr = np.tile(sigma,(x_len,1))
    gamma_arr = np.tile(gamma,(x_len,1))
    x_arr = np.tile(x,(param_len,1)).T
    #print(A_arr.shape,x_arr.shape)
    #print(A_arr.shape,mu_arr.shape,sigma_arr.shape,x_arr.shape)
    gauss = 1/(sigma_arr*math.sqrt(2*math.pi))*np.exp(-(x_arr-mu_arr)**2/(2*sigma_arr**2))
    lorentz = 1/math.pi*(gamma_arr/2)/((x_arr-mu_arr)**2+(gamma_arr/2)**2)
    summed = gauss+lorentz
    return A_arr*summed/np.max(summed)

def resids(params,x,data,uncert):
        params_arr = np.reshape(params,(-1,4))
        model = np.sum(vect_voigt(x,params_arr[:,0],params_arr[:,1],params_arr[:,2],params_arr[:,3]), axis=1)
        return (data-model)/uncert

def midpoints(x):
    return (x[:-1]+x[1:])/2

class Window(pg.QtWidgets.QMainWindow):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.binsize = str(default_binsize)
        self.xrange = default_xrange
        self.prom = str(default_peak_prom)
        self.sigma = str(default_sigma)
        self.gamma = str(default_gamma)
        self.data = data

        pg.setConfigOptions(antialias=True)
        self.setWindowTitle("PyQtGraph")
        self.init_UiComponents()
        self.show()
    
    def bin_data(self):
        e_bin_edges = np.arange(self.xrange[0],self.xrange[1],float(self.binsize))
        self.binned_counts,_ = np.histogram(self.data[:,0],bins=e_bin_edges)
        self.energies = midpoints(e_bin_edges)
    
    def init_peak_find(self):
        peak_inds,_ = find_peaks(self.binned_counts,prominence=float(self.prom_input.text()))
        self.peak_energies = self.energies[peak_inds]
        self.peak_amp = self.binned_counts[peak_inds]
        self.peak_sigma = np.ones(len(self.peak_energies))*float(self.sigma)
        self.peak_gamma = np.ones(len(self.peak_energies))*float(self.gamma)

    def peak_find(self):
        peak_inds,_ = find_peaks(self.binned_counts,prominence=float(self.prom_input.text()))
        self.peak_energies = self.energies[peak_inds]
        self.peak_amp = self.binned_counts[peak_inds]
        self.peak_sigma = np.ones(len(self.peak_energies))*float(self.sigma)
        self.peak_gamma = np.ones(len(self.peak_energies))*float(self.gamma)
        self.vals = np.column_stack((self.peak_energies,self.peak_amp,self.peak_sigma,self.peak_gamma))
        self.table.setData(self.vals)
        self.update_table()
    
    def process_table_item(self, item):
        item = item.text()
        try:
            item = float(item)
        except:
            item = np.NaN
        return item

    def read_table(self):
        tab_values = np.array([self.process_table_item(item) for item in self.table.items])
        return np.column_stack((tab_values[::4],tab_values[1::4],tab_values[2::4],tab_values[3::4]))

    def update_table(self):
        self.curr_vals = self.read_table()
        if not np.array_equal(self.vals,self.curr_vals):
            self.prev_vals = self.vals
        self.vals = copy.deepcopy(self.curr_vals)
        self.vals = self.vals[~np.isnan(self.vals).any(axis=1)]
        self.table.clear()
        self.table.setData(self.vals)

        self.peak_energies = self.vals[:,0]
        self.peak_amp = self.vals[:,1]
        self.peak_sigma = self.vals[:,2]
        self.peak_gamma = self.vals[:,3]
        self.update_plot()

    def update_table_wvals(self):
        if not np.array_equal(self.vals,self.curr_vals):
            self.prev_vals = self.vals
        self.vals = copy.deepcopy(self.curr_vals)
        self.vals = self.vals[~np.isnan(self.vals).any(axis=1)]
        self.table.clear()
        self.table.setData(self.vals)

        self.peak_energies = self.vals[:,0]
        self.peak_amp = self.vals[:,1]
        self.peak_sigma = self.vals[:,2]
        self.peak_gamma = self.vals[:,3]
        self.update_plot()

    def keyPressEvent(self, ev):
        self.sigKeyPress.emit(ev)

    def mouse_clicked(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            vb = self.pw.plotItem.vb
            scene_coords = ev.scenePos()
            if self.pw.sceneBoundingRect().contains(scene_coords):
                mouse_point = vb.mapSceneToView(scene_coords)
                self.table.addRow([mouse_point.x(),mouse_point.y(),self.sigma,self.gamma])
                self.update_table()
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            vb = self.pw.plotItem.vb
            scene_coords = ev.scenePos()
            if self.pw.sceneBoundingRect().contains(scene_coords):
                mouse_point = vb.mapSceneToView(scene_coords)
                self.curr_vals = copy.deepcopy(self.vals)
                self.curr_vals[np.argmin(abs(mouse_point.x()-self.vals[:,0])),0] = np.NAN
                self.update_table_wvals()

    def update_plot(self):
        self.pt.clear()
        self.pt.plot(self.energies,self.binned_counts,pen=pg.mkPen('w', width=2))
        self.pt.multiDataPlot(x=self.energies,y=vect_voigt(self.energies,self.peak_amp,self.peak_energies,self.peak_sigma,self.peak_gamma).T,constKwargs={'pen':pg.mkPen('g', width=1)})
        self.draw_region()

    def update_voigts(self):
        self.gamma = self.gamma_input.text()
        self.sigma = self.sigma_input.text()
        self.curr_vals = copy.deepcopy(self.vals)
        self.curr_vals[:,2] = np.ones(len(self.peak_energies))*float(self.sigma)
        self.curr_vals[:,3] = np.ones(len(self.peak_energies))*float(self.gamma)
        self.update_table_wvals()

    def undo(self):
        self.table.clear()
        self.table.setData(self.prev_vals)
        self.update_table()

    def region_toggle(self):
        self.display_region = not self.display_region
        self.update_plot()

    def draw_region(self):
        if self.display_region:
            self.lr = pg.LinearRegionItem(self.region_bounds,brush=(150,150,150,10))
            self.lr.setZValue(-10)
            self.pt.addItem(self.lr)
            self.lr.sigRegionChangeFinished.connect(self.update_bounds)

    def update_bounds(self):
        self.region_bounds = self.lr.getRegion()

    def run_fit(self):
        if self.display_region:
            fit_energies_mask = (self.energies>=self.region_bounds[0])&(self.energies<=self.region_bounds[1])
            fit_energies = self.energies[fit_energies_mask]
            fit_counts = self.binned_counts[fit_energies_mask]

            fit_peaks_mask = (self.peak_energies>=self.region_bounds[0])&(self.peak_energies<=self.region_bounds[1])
            fit_peak_energies = self.peak_energies[fit_peaks_mask]
            fit_peak_amp = self.peak_amp[fit_peaks_mask]
        else:
            fit_energies = self.energies
            fit_counts = self.binned_counts
            fit_peak_energies = self.peak_energies
            fit_peak_amp = self.peak_amp

        params_tot = Parameters()
        for en, amp in zip(fit_peak_energies, fit_peak_amp):
            to_add = [(f'P{en*10:.0f}_A', float(amp*3), True, 0, None, None, None),
                      (f'P{en*10:.0f}_mu', float(en), True, float(en)-5, float(en)+5, None, None),
                      (f'P{en*10:.0f}_sigma', float(self.sigma), not self.fix_sigma.isChecked(), 0, None, None, None),
                      (f'P{en*10:.0f}_gamma', float(self.gamma), not self.fix_gamma.isChecked(), 0, None, None, None)]
            params_tot.add_many(*to_add)

        uncert = fit_counts**.5
        uncert[uncert==0] = 1e-6
        self.results = minimize(resids,params_tot,args=(fit_energies, fit_counts, uncert))
        fit_params = np.array(list(self.results.params.valuesdict().values()))
        fit_params= np.reshape(fit_params,(-1,4))
        self.curr_vals = fit_params[:,(1,0,2,3)]
        fit_curve = np.sum(vect_voigt(fit_energies,fit_params[:,0],fit_params[:,1],fit_params[:,2],fit_params[:,3]), axis=1)
        self.update_table_wvals()
        self.pt.plot(fit_energies, fit_curve, pen=pg.mkPen('r', width=1))

    def rebin(self):
        self.binsize = float(self.binsize_input.text())
        e_bin_edges = np.arange(self.xrange[0],self.xrange[1],self.binsize)
        self.binned_counts,_ = np.histogram(self.data[:,0],bins=e_bin_edges)
        self.energies = midpoints(e_bin_edges)
        self.peak_find()

    def save_fit(self):
        print(fit_report(self.results))
        with open('fit.txt', 'a') as the_file:
            for param in self.results.params.keys():
                if re.search(r'_mu',param):
                    the_file.write(f'{self.results.params[param].value:.1f},{self.results.params[param].stderr:.1f}\n')


    def init_table_widget(self):
        undo_btn = QtWidgets.QPushButton('Undo Last Action')
        pbtn = QtWidgets.QPushButton('Print')
        pdbtn = QtWidgets.QPushButton('Find Peaks')
        rebin_btn = QtWidgets.QPushButton('ReBin Data')

        self.binsize_input = QtWidgets.QLineEdit()
        self.binsize_input.setValidator(QtGui.QDoubleValidator())
        self.binsize_input.insert(self.binsize)

        self.prom_input = QtWidgets.QLineEdit()
        self.prom_input.setValidator(QtGui.QDoubleValidator())
        self.prom_input.insert(self.prom)

        self.table = pg.TableWidget(editable=True)

        undo_btn.clicked.connect(self.undo)
        pbtn.clicked.connect(lambda: print(self.read_table()))
        self.sigKeyPress.connect(lambda event: self.update_table() if event.key()==16777220 else False)
        pdbtn.clicked.connect(self.peak_find)
        rebin_btn.clicked.connect(self.rebin)

        self.table.setFormat('%.2f')
        self.bin_data()
        self.init_peak_find()
        self.vals = np.column_stack((self.peak_energies,self.peak_amp,self.peak_sigma,self.peak_gamma))
        self.prev_vals = self.vals
        self.table.setData(self.vals)

        lo = pg.LayoutWidget()
        lo.addWidget(self.table,1,0,1,3)

        lo.addWidget(QtWidgets.QLabel('Bin Size='),2,0,1,1)
        lo.addWidget(self.binsize_input,2,1)
        lo.addWidget(rebin_btn,2,2,1,1)

        lo.addWidget(QtWidgets.QLabel('Peak Prom='),3,0,1,1)
        lo.addWidget(self.prom_input,3,1)
        lo.addWidget(pdbtn,3,2,1,1)

        lo.addWidget(undo_btn,6,0,1,3)
        #lo.addWidget(pbtn,7,0)
        dock = Dock('',size=(100,100))
        dock.addWidget(lo)
        lo.setMinimumSize(1,1)

        return dock

    def init_voigt_widget(self):
        self.gamma_input = QtWidgets.QLineEdit()
        self.gamma_input.setValidator(QtGui.QDoubleValidator())
        self.gamma_input.insert(self.gamma)

        self.sigma_input = QtWidgets.QLineEdit()
        self.sigma_input.setValidator(QtGui.QDoubleValidator())
        self.sigma_input.insert(self.sigma)

        update_btn = QtWidgets.QPushButton('Update Widths')
        update_btn.clicked.connect(self.update_voigts)

        lo = pg.LayoutWidget()
        lo.addWidget(QtWidgets.QLabel('Voigt Parameters:'),1,0,1,4)
        lo.addWidget(QtWidgets.QLabel('Gamma='),2,0,1,1)
        lo.addWidget(self.gamma_input,2,1,1,1)
        lo.addWidget(QtWidgets.QLabel('Sigma='),2,2,1,1)
        lo.addWidget(self.sigma_input,2,3,1,1)
        lo.addWidget(update_btn,3,0,1,4)
        dock = Dock('',size=(100,10))
        lo.setMinimumSize(1,1)
        dock.addWidget(lo)

        return dock
    
    def init_plot_widget(self):
        self.pw = pg.PlotWidget(show=True,enableMenu=False)
        self.pt = self.pw.plotItem
        self.update_plot()
        self.pt.scene().sigMouseClicked.connect(self.mouse_clicked)

        lo = pg.LayoutWidget()
        lo.addWidget(self.pw,0,1)
        dock = Dock('',size=(800,100))
        dock.addWidget(lo)

        return dock
    
    def init_fitting_widget(self):
        self.region_bounds = [np.min(self.peak_energies),np.max(self.peak_energies)]
        self.display_region = False
        toggle_region_btn = QtWidgets.QPushButton('Toggle Fit Region')
        fit_btn = QtWidgets.QPushButton('Run Fit')
        results_btn = QtWidgets.QPushButton('Save and Display Results')
        self.fix_sigma = QtWidgets.QCheckBox('Fix Sigma')
        self.fix_sigma.setChecked(True)
        self.fix_gamma = QtWidgets.QCheckBox('Fix Gamma')
        self.fix_gamma.setChecked(True)

        toggle_region_btn.clicked.connect(self.region_toggle)
        fit_btn.clicked.connect(self.run_fit)
        results_btn.clicked.connect(self.save_fit)

        lo = pg.LayoutWidget()
        lo.addWidget(self.fix_sigma,0,0,1,1)
        lo.addWidget(self.fix_gamma,0,1,1,1)
        lo.addWidget(toggle_region_btn,1,0,1,2)
        lo.addWidget(fit_btn,2,0,1,2)
        lo.addWidget(results_btn,3,0,1,2)
        dock = Dock('',size=(100,15))
        dock.addWidget(lo)
        lo.setMinimumSize(1,1)

        return dock

    def init_UiComponents(self):
        widget = QtWidgets.QWidget()

        voigt_dock = self.init_voigt_widget()
        table_dock = self.init_table_widget()
        fitting_dock = self.init_fitting_widget()
        plot_dock = self.init_plot_widget()
        

        area = DockArea()
        self.setCentralWidget(area)
        area.addDock(table_dock,'left')
        area.addDock(plot_dock,'right')
        area.addDock(voigt_dock,'top',table_dock)
        area.addDock(fitting_dock,'bottom',table_dock)
        self.showMaximized()

app = pg.mkQApp("Plotting Example")
window = Window()
pg.exec()