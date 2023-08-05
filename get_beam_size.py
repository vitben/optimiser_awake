"""
GET BEAM SIZE
R. Ramjiawan
Oct 2020
Track beam through line and extract beam parameters at merge-point
"""

import numpy as np
import scipy as scp
from scipy import stats

class getBeamSize:
    def __init__(self, q, n_particles, madx, init_dist, foil_w, x):
        self.q = q
        self.x = x
        self.no_quad = sum(np.array([y['type'] for y in x.values()]) == 'quadrupole')
        self.no_sext = sum(np.array([y['type'] for y in x.values()]) == 'sextupole')
        self.no_oct = sum(np.array([y['type'] for y in x.values()]) == 'octupole')
        self.no_dist = sum(np.array([y['type'] for y in x.values()]) == 'distance')
        self.n_particles = n_particles
        self.madx = madx
        self.verbose = True
        self.ptc = False
        self.name = [y['name'] for y in x.values()]
        self.init_dist = init_dist
        x0 = self.init_dist[:, 0]
        px0 = self.init_dist[:, 3]
        y0 = self.init_dist[:, 1]
        py0 = self.init_dist[:, 4]
        self.foil_w = foil_w
        self.dof = self.no_quad + self.no_sext + self.no_oct + self.no_dist
        self.gamma = np.sqrt(0.511e-3 ** 2 + 150e-3 ** 2) / 0.511e-3
        self.beta_nom = 10**3 * (10**6 * np.divide(2*8.854187817*(10**-12)*(9.10938356*10**-31)*((3*10**8)**2)*self.gamma, (7*10**14)*(1.6*10**-19)**2))**0.25
        
        # Calculate bunch parameters from input distribution
        self.emitx_before = self.calcEmit(x0, px0)
        self.emity_before = self.calcEmit(y0, py0)
        self.betx0 = np.divide(np.mean(np.multiply(x0, x0)), self.emitx_before)
        self.alfx0 = -np.divide(np.mean(np.multiply(x0, px0)), self.emitx_before)
        self.bety0 = np.divide(np.mean(np.multiply(y0, y0)), self.emity_before)
        self.alfy0 = -np.divide(np.mean(np.multiply(y0, py0)), self.emity_before)
        self.sig_nom_x = np.multiply(np.sqrt(self.emitx_before), self.beta_nom)
        self.sig_nom_y = np.multiply(np.sqrt(self.emity_before), self.beta_nom)
        print("input emittance x = {:.4f}, {:.4f} um".format(self.emitx_before * self.gamma * 10 ** 6,
                                                             self.emity_before * self.gamma * 10 ** 6))
        print("beta - x,y = {:.4f}, {:.4f}; alpha - x,y = {:.4f}, {:.4f}: ".format(self.betx0, self.bety0, self.alfx0,
                                                                                   self.alfy0))

    def get_beam_size_twiss(self):
        beam_size_x_waist = 100
        beam_size_y_waist = 100
        loss, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, _, kl_divergence, \
        sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
        error_flag = 0
        try:
            self.set_quads()
            self.madx.use(sequence='TT43', range="TT43$START/TT43$END")
            self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                            DY=0, dpy=0)
            ptc_output, error_flag = self.track('TT43$START', 'TT43$END', ['MERGE', 'WAIST'], self.init_dist)
        except RuntimeError:
            error_flag = 1
        if error_flag == 0:
            x0 = np.array(ptc_output['x'].loc['merge'])
            y0 = np.array(ptc_output['y'].loc['merge'])
            z0 = np.array(ptc_output['t'].loc['merge'])
            px0 = np.array(ptc_output['px'].loc['merge'])
            py0 = np.array(ptc_output['py'].loc['merge'])

            all_u0 = x0 + y0 + z0 + px0 + py0
            x0 = x0[~np.isnan(all_u0)]
            y0 = y0[~np.isnan(all_u0)]
            z0 = z0[~np.isnan(all_u0)]
            px0 = px0[~np.isnan(all_u0)]
            py0 = py0[~np.isnan(all_u0)]
            loss = self.n_particles - len(x0)
            if loss > 430:
                _, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, _, kl_divergence, \
                sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
            else:
                emitx = self.calcEmit(x0, px0)
                sig_nom_x_after = np.multiply(np.sqrt(emitx), self.beta_nom)
                emity = self.calcEmit(y0, py0)
                sig_nom_y_after = np.multiply(np.sqrt(emity), self.beta_nom)

                print("output emittance x, y = {:.4f}, {:.4f} um".format(emitx * self.gamma * 10 ** 6,
                                                                         emity * self.gamma * 10 ** 6))

                beam_size_x, beam_size_y, beam_size_z, beam_pos_x, beam_pos_y, frac_x, frac_y = self.get_beam_params(x0, y0, z0)

                # Calculate the KL divergence
                kl_divergence_x = self.calcKL(x0)
                kl_divergence_y = self.calcKL(y0)
                kl_divergence_px = self.calcKL(px0)
                kl_divergence_py = self.calcKL(py0)
                kl_divergence = np.sqrt(np.square(kl_divergence_x) + np.square(kl_divergence_y)
                                        + 0*np.square(kl_divergence_px) + 0*np.square(kl_divergence_py))
                print("KL divergences = {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(kl_divergence_x, kl_divergence_y,
                                                                               kl_divergence_px, kl_divergence_py))

                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0)
                if 'merge:1' in twiss['name']:
                    betx = twiss['betx'][twiss['name'] == 'merge:1'][0]
                    bety = twiss['bety'][twiss['name'] == 'merge:1'][0]
                    alfx = twiss['alfx'][twiss['name'] == 'merge:1'][0]
                    alfy = twiss['alfy'][twiss['name'] == 'merge:1'][0]
                    dx = twiss['dx'][twiss['name'][:] == 'merge:1'][0]
                    dx2 = twiss['dx'][twiss['name'][:] == 'tt43$end:1'][0]

                    self.del_madx_tab()
                    if betx < 0 or bety < 0:
                        betx = 100
                        bety = 100
                else:
                    betx = 100
                    bety = 100
                maxbety = 10000
                maxbetx = 10000
                print("beta - x,y = {:.4f}, {:.4f}; alpha - x,y = {:.4f}, {:.4f}; "
                      "dx - inj, end = {:.4f}, {:.4f}): ".format(betx, bety, alfx, alfy, dx, dx2))
        return beam_size_x, beam_size_y, beam_size_z, loss, error_flag, alfx, alfy, kl_divergence, dx, dx2, self.sig_nom_x, self.sig_nom_y, sig_nom_x_after, sig_nom_y_after, float(
            maxbetx), float(maxbety), beam_pos_x, beam_pos_y, frac_x, frac_y

    def get_beam_size(self):
        _, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, _, kl_divergence, \
        sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
        # set magnet strengths/positions
        self.set_quads()
        try:
            ptc_output, error_flag = self.ptc_track('TT43$START', 'TT43$END', ['MERGE'], self.init_dist)
            twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                    DY=0, dpy=0)
            x0 = np.array(ptc_output['x'].loc['merge'])
            y0 = np.array(ptc_output['y'].loc['merge'])
            z0 = np.array(ptc_output['t'].loc['merge'])
            px0 = np.array(ptc_output['px'].loc['merge'])
            py0 = np.array(ptc_output['py'].loc['merge'])
            loss = self.n_particles - len(np.atleast_1d(x0))
        except RuntimeError:
            error_flag = 1
        if error_flag == 1:
            loss = self.n_particles
        else:

            # Heavily penalise the loss of too many particle
            if loss > 0.8 * self.n_particles:
                print("loss is at " + str(100 * loss / self.n_particles) + "%")
                _, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, _, kl_divergence, \
                sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
            else:
                emitx = self.calcEmit(x0, px0)
                print("output emittance x = {:.4f} um".format(emitx * self.gamma * 10 ** 6))
                sig_nom_x_after = np.multiply(np.sqrt(emitx), self.beta_nom)

                emity = self.calcEmit(y0, py0)
                print("output emittance y = {:.4f} um".format(emity * self.gamma * 10 ** 6))
                sig_nom_y_after = np.multiply(np.sqrt(emity), self.beta_nom)

                beam_size_x, beam_size_y, beam_size_z, beam_pos_x, beam_pos_y, frac_x, frac_y = self.get_beam_params(x0, y0, z0)
                # Calculate the KL divergence
                kl_divergence_x = self.calcKL(x0)
                kl_divergence_y = self.calcKL(y0)
                kl_divergence_px = self.calcKL(px0)
                kl_divergence_py = self.calcKL(py0)
                kl_divergence = np.sqrt(np.square(kl_divergence_x) + np.square(kl_divergence_y)
                                        + 0*np.square(kl_divergence_px) + 0*np.square(kl_divergence_py))
                if self.verbose:
                    print("KL divergences = {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(kl_divergence_x, kl_divergence_y,
                                                                                   kl_divergence_px, kl_divergence_py))

                ptc_twiss = self.ptc_twiss()
                # Calculate twiss parameters at merge-point
                if 'merge:1' in ptc_twiss['name']:
                    betx = twiss['betx'][ptc_twiss['name'] == 'merge:1'][0]
                    bety = twiss['bety'][ptc_twiss['name'] == 'merge:1'][0]

                    alfx = ptc_twiss['alfx'][ptc_twiss['name'] == 'merge:1'][0]
                    alfy = ptc_twiss['alfy'][ptc_twiss['name'] == 'merge:1'][0]

                    dx = twiss['dx'][twiss['name'][:] == 'merge:1'][0]
                    dx2 = twiss['dx'][twiss['name'][:] == 'tt43$end:1'][0]

                    if betx < 0 or bety < 0:
                        betx = 100
                        bety = 100
                else:  # if beam lost before merge
                    betx = 100
                    bety = 100
                    alfx = 1
                    alfy = 1
                    dx = 1
                    dx2 = 1
                maxbety = 10000
                maxbetx = 10000
                print("beta - x,y = {:.4f}, {:.4f}; alpha - x,y = {:.4f}, {:.4f}; "
                      "dx - inj, end = {:.4f}, {:.4f}): ".format(betx, bety, alfx, alfy, dx, dx2))
        return beam_size_x, beam_size_y, beam_size_z, loss, error_flag, alfx, alfy, kl_divergence, dx, dx2, self.sig_nom_x, self.sig_nom_y, sig_nom_x_after, sig_nom_y_after, float(
            maxbetx), float(maxbety), beam_pos_x, beam_pos_y, frac_x, frac_y

    def get_beam_size_foil(self):
        self.ptc = True


        loss, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, _, kl_divergence, \
        sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
        betx = 1000
        bety = 1000
        try:
            self.set_quads()
            self.madx.use(sequence='TT43', range="TT43$START/TT43$END")
            self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                            DY=0, dpy=0)
            ptc_output, error_flag = self.ptc_track('TT43$START', 'TT43$END', ['MERGE', 'FOIL1'], self.init_dist)
        except RuntimeError:
            print("RuntimeError")
            error_flag = 1
        if error_flag == 1:
            loss = self.n_particles
        else:
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
            e_beta = 0.999995
            e_mom = 150
            rad_length = 8.9e-2

            if self.foil_w == 0:
                theta_0 = 0
            else:
                theta_temp = 13.6 / (e_beta * e_mom) / np.sqrt(rad_length)
                theta_0 = theta_temp * np.sqrt(self.foil_w) * (1. + 0.038 * np.log(self.foil_w / rad_length))
            if 'foil1' in ptc_output.index:
                x0 = np.array(ptc_output['x'].loc['foil1'])
                y0 = np.array(ptc_output['y'].loc['foil1'])
                t0 = np.array(ptc_output['t'].loc['foil1'])
                np.random.seed(42)
                px0 = np.array(ptc_output['px'].loc['foil1']) + theta_0 * np.random.normal(size=np.shape(x0))
                np.random.seed(21)
                py0 = np.array(ptc_output['py'].loc['foil1']) + theta_0 * np.random.normal(size=np.shape(y0))
                pt0 = np.array(ptc_output['pt'].loc['foil1'])
                all_u0 = x0 + y0 + t0 + px0 + py0
                x0 = x0[~np.isnan(all_u0)]
                y0 = y0[~np.isnan(all_u0)]
                t0 = t0[~np.isnan(all_u0)]
                px0 = px0[~np.isnan(all_u0)]
                py0 = py0[~np.isnan(all_u0)]
                pt0 = pt0[~np.isnan(all_u0)]
                self.del_madx_tab()
                init_dist = np.transpose([x0, y0, t0, px0, py0, pt0])
                print(len(init_dist))
                if len(init_dist) > 12:
                    ptc_output, error_flag = self.ptc_track('FOIL1', 'TT43$END', ['MERGE', 'FOIL2'], init_dist)
                    if error_flag == 0:
                        if 'foil2' in ptc_output.index:
                            x0 = np.array(ptc_output['x'].loc['foil2'])
                            y0 = np.array(ptc_output['y'].loc['foil2'])
                            t0 = np.array(ptc_output['t'].loc['foil2'])
                            np.random.seed(100)
                            px0 = np.array(ptc_output['px'].loc['foil2']) + theta_0 * np.random.normal(
                                size=np.shape(x0))
                            np.random.seed(200)
                            py0 = np.array(ptc_output['py'].loc['foil2']) + theta_0 * np.random.normal(
                                size=np.shape(y0))
                            pt0 = np.array(ptc_output['pt'].loc['foil2'])
                            all_u0 = x0 + y0 + t0 + px0 + py0
                            x0 = x0[~np.isnan(all_u0)]
                            y0 = y0[~np.isnan(all_u0)]
                            t0 = t0[~np.isnan(all_u0)]
                            px0 = px0[~np.isnan(all_u0)]
                            py0 = py0[~np.isnan(all_u0)]
                            pt0 = pt0[~np.isnan(all_u0)]
                            self.del_madx_tab()
                            init_dist = np.transpose([x0, y0, t0, px0, py0, pt0])
                            print(len(init_dist))
                            if len(init_dist) > 12:
                                ptc_output, error_flag = self.ptc_track('FOIL2', 'TT43$END', ['MERGE'],
                                                                        init_dist)
                                if error_flag == 0:
                                    x0 = np.array(ptc_output['x'].loc['merge'])
                                    y0 = np.array(ptc_output['y'].loc['merge'])
                                    z0 = np.array(ptc_output['t'].loc['merge'])
                                    px0 = np.array(ptc_output['px'].loc['merge'])
                                    py0 = np.array(ptc_output['py'].loc['merge'])


                        if self.ptc == False:
                            all_u0 = x0 + y0 + z0 + px0 + py0
                            x0 = x0[~np.isnan(all_u0)]
                            y0 = y0[~np.isnan(all_u0)]
                            z0 = z0[~np.isnan(all_u0)]
                            px0 = px0[~np.isnan(all_u0)]
                            py0 = py0[~np.isnan(all_u0)]
                        loss = self.n_particles - len(x0)

                        if loss > 0.5 * self.n_particles:
                            print("loss is at " + str(100 * loss / self.n_particles) + "%")
                        else:
                            emitx = self.calcEmit(x0, px0)
                            self.sig_nom_x = np.multiply(np.sqrt(emitx), self.beta_nom)
                            emity = self.calcEmit(y0, py0)
                            self.sig_nom_y = np.multiply(np.sqrt(emity), self.beta_nom)

                            print("output emittance x, y = {:.4f}, {:.4f} um".format(emitx * self.gamma * 10 ** 6,
                                                                                     emity * self.gamma * 10 ** 6))
                            beam_size_x, beam_size_y, beam_size_z, beam_pos_x, beam_pos_y, frac_x, frac_y = self.get_beam_params(x0, y0, z0)

                            betx = np.divide(np.mean(np.multiply(x0, x0)), emitx)
                            alfx = -np.divide(np.mean(np.multiply(x0, px0)), emitx)
                            bety = np.divide(np.mean(np.multiply(y0, y0)), emity)
                            alfy = -np.divide(np.mean(np.multiply(y0, py0)), emity)

                            # Calculate the KL divergence
                            kl_divergence_x = self.calcKL(x0)
                            kl_divergence_y = self.calcKL(y0)
                            kl_divergence_px = self.calcKL(px0)
                            kl_divergence_py = self.calcKL(py0)
                            kl_divergence = np.sqrt(np.square(kl_divergence_x) + np.square(kl_divergence_y)
                                                    + np.square(kl_divergence_px) + np.square(kl_divergence_py))
                            print("KL divergences = {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(kl_divergence_x, kl_divergence_y,
                                                                                           kl_divergence_px, kl_divergence_py))
                            self.del_madx_tab()
                            self.madx.use(sequence='TT43', range="TT43$START/TT43$END")
            if self.ptc == True:
                ptc_twiss = self.ptc_twiss()
            else:
                ptc_twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                            DY=0, dpy=0)
            twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                    DY=0, dpy=0)
            if 'merge:1' in ptc_twiss['name']:
                betx = ptc_twiss['betx'][ptc_twiss['name'] == 'merge:1'][0]
                bety = ptc_twiss['bety'][ptc_twiss['name'] == 'merge:1'][0]
                dx = twiss['dx'][twiss['name'][:] == 'merge:1'][0]
                dx2 = twiss['dx'][twiss['name'][:] == 'tt43$end:1'][0]
                if betx < 0 or bety < 0:
                    betx = 1000
                    bety = 1000
            else:
                print("twiss unstable")
        print("beta - x,y = {:.4f}, {:.4f}; alpha - x,y = {:.4f}, {:.4f}; "
              "dx - inj, end = {:.4f}, {:.4f}): ".format(betx, bety, alfx, alfy, dx, dx2))
        return beam_size_x, beam_size_y, beam_size_z, loss, error_flag, alfx, alfy, kl_divergence, dx, dx2, self.sig_nom_x, self.sig_nom_y, sig_nom_x_after, sig_nom_y_after, betx, bety, beam_pos_x, beam_pos_y, frac_x, frac_y

    def ptc_track(self, start, end, observe, init_dist):
        error_flag = 0
        try:
            self.madx.use(sequence='TT43', range=start + "/" + end)
            self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                            DY=0, dpy=0)
            self.madx.ptc_create_universe()
            self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
            self.madx.ptc_align()
            self.madx.ptc_setswitch(fringe=True)
            for obs in observe:
                self.madx.ptc_observe(place=obs)
            self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
            self.madx.ptc_twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                                dpy=0)
            with self.madx.batch():
                for particle in init_dist:
                    self.madx.ptc_start(x=-particle[0], px=-particle[3], y=particle[1], py=particle[4],
                                        t=1 * particle[2], pt=2.09 * particle[5])
                self.madx.ptc_track(icase=56, element_by_element=True, dump=False, onetable=True, recloss=True,
                                    maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
                self.madx.ptc_track_end()
            ptc_output = self.madx.table.trackone
            ptc_output = ptc_output.dframe()

        except RuntimeError or IndexError:  # If magnets overlap or twiss incomputable
            print('MAD-X Error occurred, re-spawning MAD-X process')
            error_flag = 1
            ptc_output = {}
        return ptc_output, error_flag

    def track(self, start, end, observe, init_dist):
        ptc_output = {}
        error_flag = 0
        try:
            with self.madx.batch():
                self.madx.track(onetable=True, recloss=True, onepass=True)
                for particle in init_dist:
                    self.madx.start(x=-particle[0], px=-particle[3], y=particle[1], py=particle[4],
                                    t=1 * particle[2], pt= 2.09*particle[5])
                for obs in observe:
                    self.madx.observe(place=obs)
                self.madx.run(turns=1, maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
                self.madx.endtrack()

        except RuntimeError:
            print('MAD-X Error occurred, re-spawning MAD-X process')
            loss, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, error_flag, kl_divergence, \
            sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
        else:
            ptc_output = self.madx.table.trackone
            ptc_output = ptc_output.dframe()
        return ptc_output, error_flag

    def set_quads(self):
        for j in range(self.dof):
            self.madx.input(self.name[j] + "=" + str(self.q[j]) + ";")
            # if self.verbose:
            print(self.name[j][0] + self.name[j][-1] + "=" + str(self.q[j]))
        self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                        DY=0, dpy=0)

    @staticmethod
    def calcEmit(u0, pu0):
        temp = np.mean((u0 - np.mean(u0)) * (pu0 - np.mean(pu0)))
        emit = np.sqrt(np.std(u0) ** 2 * np.std(pu0) ** 2 - temp ** 2)
        return emit

    @staticmethod
    def calcKL(u0):
        std = (np.std(u0))
        pk = scp.stats.norm(0, std).pdf(np.linspace(-4 * std, 4 * std, num=50))
        qk_temp = stats.gaussian_kde(np.array(u0))
        qk = qk_temp([np.linspace(-4 * std, 4 * std, num=50)])
        kl_div = stats.entropy(pk=pk, qk=qk)
        if kl_div == float("inf"):
            kl_div = 10
        return kl_div

    def get_beam_params(self, x0, y0, z0, rms=False):
        if rms:
            beam_size_x = np.multiply(self.rmsValue(x0), 10 ** 6)
            beam_size_y = np.multiply(self.rmsValue(y0), 10 ** 6)
            beam_size_z = np.multiply(self.rmsValue(z0), 10 ** 6)
        else:
            beam_size_x = np.multiply(np.std(x0), 10 ** 6)
            beam_size_y = np.multiply(np.std(y0), 10 ** 6)
            beam_size_z = np.multiply(np.std(z0), 10 ** 6)
        beam_pos_x = np.multiply(np.mean(x0), 10 ** 6)
        beam_pos_y = np.multiply(np.mean(y0), 10 ** 6)
        # frac_x = np.divide(sum(abs(x0)<5.76*10**-6), len(x0))
        # frac_y = np.divide(sum(abs(y0)<5.76*10**-6), len(y0))
        frac_x = stats.kurtosis(self.reject_outliers(x0), fisher=True)
        frac_y = stats.kurtosis(self.reject_outliers(y0), fisher=True)
        return beam_size_x, beam_size_y, beam_size_z, beam_pos_x, beam_pos_y, frac_x, frac_y

    def reject_outliers(self, data):
        ind = abs(data - np.mean(data)) < 5 * np.std(data)
        return data[ind]


    def penalise(self):
        loss = self.n_particles
        frac_x = 1e8
        frac_y = 1e8
        beam_size_x = 1e8
        beam_size_y = 1e8
        beam_pos_x = 1e3
        beam_pos_y = 1e3
        beam_size_z = 1e3
        error_flag = 1
        kl_divergence = 1
        sig_nom_x_after = 0
        sig_nom_y_after = 0
        maxbetx = 1
        maxbety = 1
        alfx = 10
        alfy = 10
        dx = 1
        dx2 = 1
        return loss, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, error_flag, kl_divergence, \
               sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y

    def rmsValue(self, arr):
        n = len(arr)
        square = 0
        for i in range(0, n):
            square += (arr[i] ** 2)
        mean = (square / float(n))
        root = np.sqrt(mean)
        return root

    def del_madx_tab(self):
        self.madx.input("delete, table = trackone;")
        self.madx.input("delete, table = trackloss;")
        self.madx.input("delete, table = tracksumm;")

    def ptc_twiss(self):
        self.madx.ptc_create_universe()
        self.madx.ptc_align()
        self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.ptc_twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                            dpy=0, no=5)
        twiss = self.madx.table['ptc_twiss']
        return twiss
