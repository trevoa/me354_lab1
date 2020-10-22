import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import sin, cos, pi
from itertools import product

E = 69000e6  # Elastic Modulus
v = 0.33     # Poisson's Ratio
Y = 275e6    # Yield Strength

C = np.array([[1/E, -v/E, 0], [-v/E, 1/E, 0], [0, 0, 2*(1+v)/E]])    # Stiffness tensor

# Max and min values for each dimension
l = [0.69992, 0.70008]
b = [0.038325, 0.038375]
h = [0.025715, 0.025765]
a = [0.2]
g1x = [0.34992, 0.35008]
g2x = [0.37492, 0.37508]
g2z = [0.01265, 0.01285]
g3x = [0.34992, 0.35008]
g4x = [0.44992, 0.45008]

ros1 = [0, 45, 90]
ros2 = [-45, 0, 45]


def strain1Rotate(epsilon,th):
    '''Rotate a strain epsilon =[e_x,e_y,gamma_xy] by an angle th given in radians.'''
    T = [cos(th)**2, sin(th)**2, sin(th)*cos(th)]
    return np.dot(T, epsilon)


class ThreePointBeam:

    def __init__(self, length, base, height, g1x, g2x, g2z, g3x, g4x, load):
        self.l = length    # length in m
        self.b = base      # base in m
        self.h = height    # height in m
        self.p = load      # load in N
        self.g1x = g1x
        self.g2x = g2x
        self.g2z = g2z
        self.g3x = g3x
        self.g4x = g4x

    # Calculate moment of inertia
    def areaMomentOfInertia(self):
        I = self.b * self.h ** 3 / 12
        return I

    # Max deflection for 3-point bend
    def Deflection(self):
        w = self.p * self.l ** 3 / (48 * E * self.areaMomentOfInertia())
        return w

    # Shear force for 3-point bending
    def shearForce(self, x):
        if x <= self.l / 2:
            Q = self.p / 2
        else:
            Q = -self.p / 2
        return Q

    # Moment M for 3-points
    def bendingMoment(self, x):
        if x <= self.l / 2:
            M = self.p * x / 2
        else:
            M = self.p * (self.l - x) / 2
        return M

    # Axial stress by moment
    def axialStress(self, x, z):
        sigma = self.bendingMoment(x) * z / self.areaMomentOfInertia()
        sigma *= 1e-6
        return sigma

    def ShearStress(self, x, z):
        S = (self.shearForce(x) / 2 * self.areaMomentOfInertia()) * ((self.h ** 2 / 4) - z ** 2)
        S *= 1e-6
        return S

    def strainReadings(self, x, z, ros_angles):
        sig = self.axialStress(x, z) * 1e6
        tau = self.ShearStress(x, z) * 1e6
        epsilon = np.dot(C, np.array([sig, 0, tau]))
        epsilon_rosette = np.array([strain1Rotate(epsilon, th * pi / 180) for th in ros_angles])
        epsilon_rosette *= 1e6
        return epsilon_rosette


class FourPointBeam:

    def __init__(self, length, base, height, a, g1x, g2x, g2z, g3x, g4x, load):
        self.l = length    # length in m
        self.b = base      # base in m
        self.h = height    # height in m
        self.p = load      # load in N
        self.a = a
        self.g1x = g1x
        self.g2x = g2x
        self.g2z = g2z
        self.g3x = g3x
        self.g4x = g4x

    # Calculate moment of inertia
    def areaMomentOfInertia(self):
        I = self.b * self.h ** 3 / 12
        return I

    # Calculate max deflection
    def Deflection(self):
        w = ((self.p * self.a)/(48 * E * self.areaMomentOfInertia())) * ((3 * self.l**2) - (4 * self.a**2))
        return w

    def shearForce(self, x):
        if x <= self.a:
            Q = self.p / 2
        elif x > (self.l - self.a):
            Q = -self.p / 2
        else:
            Q = 0
        return Q

    def bendingMoment(self, x):
        if x <= self.a:
            M = (self.p * x) / 2
        elif x > (self.l - self.a):
            M = (self.p * (x - self.l)) / 2
        else:
            M = (self.p * self.a) / 2
        return M

    # Axial stress by moment
    def axialStress(self, x, z):
        sigma = self.bendingMoment(x) * z / self.areaMomentOfInertia()
        sigma *= 1e-6
        return sigma

    def ShearStress(self, x, z):
        S = (self.shearForce(x) / 2 * self.areaMomentOfInertia()) * ((self.h ** 2 / 4) - z ** 2)
        S *= 1e-6
        return S

    def strainReadings(self, x, z, ros_angles):
        sig = self.axialStress(x, z) * 1e6
        tau = self.ShearStress(x, z) * 1e6
        epsilon = np.dot(C, np.array([sig, 0, tau]))
        epsilon_rosette = np.array([strain1Rotate(epsilon, th * pi / 180) for th in ros_angles])
        epsilon_rosette *= 1e6
        return epsilon_rosette


def interval(min, max):
    avg = (min + max) / 2
    avg = round(avg, 3)
    error = (max - min) / 2
    error = round(error, 3)
    value = "{} +/- {}".format(avg, error)
    return value


def RunLab(beam):
    gauges = {'G1': [beam.g1x, -beam.h / 2, ros1],
              'G2': [beam.g2x, beam.g2z - (beam.h / 2), ros1],
              'G3': [beam.g3x, beam.h / 2, ros1],
              'G4': [beam.g4x, -beam.h / 2, ros2]}

    output = {'Max Deflection': beam.Deflection()}
    for gauge in gauges:
        x = gauges[gauge][0]
        z = gauges[gauge][1]
        ros = gauges[gauge][2]
        reading = list(beam.strainReadings(x, z, ros))
        shear = beam.shearForce(x)
        axial_stress = beam.axialStress(x, z)
        shear_stress = beam.ShearStress(x, z)
        moment = beam.bendingMoment(x)

        output[gauge] = {"Shear Force": shear, "Moment": moment, "Axial Stress": axial_stress,
                         "Shear Stress": shear_stress, "a": reading[0], "b": reading[1], "c": reading[2]}

    return output


combination = list(product(l, b, h, g1x, g2x, g2z, g3x, g4x))
loads = [400, 800, 1200, 1600]

print('Three Point Bending Analysis')
for load in loads:
    i = 0
    G1_list = {}
    G2_list = {}
    G3_list = {}
    G4_list = {}
    for combo in combination:
            beam = ThreePointBeam(*combo, load)
            test = RunLab(beam)
            print(test)
            G1 = test['G1']
            G2 = test['G2']
            G3 = test['G3']
            G4 = test['G4']
            G1_list[i] = G1
            G2_list[i] = G2
            G3_list[i] = G3
            G4_list[i] = G4
            i += 1

    G1_df = pd.DataFrame.from_dict(G1_list, orient='index', columns=["Shear Force", "Moment", "Axial Stress", "Shear Stress",                                                                "a", "b", "c"])
    G2_df = pd.DataFrame.from_dict(G2_list, orient='index', columns=["Shear Force", "Moment", "Axial Stress", "Shear Stress",                                                              "a", "b", "c"])
    G3_df = pd.DataFrame.from_dict(G3_list, orient='index', columns=["Shear Force", "Moment", "Axial Stress", "Shear Stress",                                                             "a", "b", "c"])
    G4_df = pd.DataFrame.from_dict(G4_list, orient='index', columns=["Shear Force", "Moment", "Axial Stress", "Shear Stress",
                                                                    "a", "b", "c"])

    df_list = [G1_df, G2_df, G3_df, G4_df]

    j = 1
    print("\n\nLoad of {}".format(load))
    for df in df_list:
        print(df)
        df.to_csv("G{}.csv".format(j))
        print("\nG{} Measurements".format(j))
        for measure in df:
            max = df[measure].max()
            min = df[measure].min()
            out = interval(min, max)
            print("{}: {}".format(measure, out))
        j += 1



combination2 = list(product(l, b, h, a, g1x, g2x, g2z, g3x, g4x))
print("\n\nFour Point Bending Analysis")
for load in loads:
    i = 0
    G1_list = {}
    G2_list = {}
    G3_list = {}
    G4_list = {}
    for combo in combination2:
            beam = FourPointBeam(*combo, load)
            test = RunLab(beam)
            G1 = test['G1']
            G2 = test['G2']
            G3 = test['G3']
            G4 = test['G4']
            G1_list[i] = G1
            G2_list[i] = G2
            G3_list[i] = G3
            G4_list[i] = G4
            i += 1

    G1_df = pd.DataFrame.from_dict(G1_list, orient='index', columns=["Shear Force", "Moment", "Axial Stress", "Shear Stress",                                                                "a", "b", "c"])
    G2_df = pd.DataFrame.from_dict(G2_list, orient='index', columns=["Shear Force", "Moment", "Axial Stress", "Shear Stress",                                                              "a", "b", "c"])
    G3_df = pd.DataFrame.from_dict(G3_list, orient='index', columns=["Shear Force", "Moment", "Axial Stress", "Shear Stress",                                                             "a", "b", "c"])
    G4_df = pd.DataFrame.from_dict(G4_list, orient='index', columns=["Shear Force", "Moment", "Axial Stress", "Shear Stress",
                                                                    "a", "b", "c"])

    df_list = [G1_df, G2_df, G3_df, G4_df]

    j = 1
    print("\n\nLoad of {}".format(load))
    for df in df_list:
        df.to_csv("G{}.csv".format(j))
        print("\nG{} Measurements".format(j))
        for measure in df:
            max = df[measure].max()
            min = df[measure].min()
            out = interval(min, max)
            print("{}: {}".format(measure, out))
        j += 1


