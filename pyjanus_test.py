import pyJanus

model_path = "models/spheres/cannonball_aero.dml"

janus = pyJanus.Janus(model_path)

referenceWingArea = janus.get_variabledef("SWING")
totalCoefficientOfLift = janus.get_variabledef("CL")
totalCoefficientOfDrag = janus.get_variabledef("CD")
aeroBodyForceCoefficient_Y = janus.get_variabledef("CY")
aeroBodyMomentCoefficient_Roll = janus.get_variabledef("Cl")
aeroBodyMomentCoefficient_Pitch = janus.get_variabledef("Cm")
aeroBodyMomentCoefficient_Yaw = janus.get_variabledef("Cn")

print(referenceWingArea)
print(referenceWingArea.get_value())
print(str(referenceWingArea.units))
print(referenceWingArea.name)

print(totalCoefficientOfLift)
print(totalCoefficientOfDrag)
print(aeroBodyForceCoefficient_Y)
print(aeroBodyMomentCoefficient_Roll)
print(aeroBodyMomentCoefficient_Pitch)
print(aeroBodyMomentCoefficient_Yaw)

print(type(referenceWingArea))

assert id(janus.get_variabledef()[0]) == id(referenceWingArea)
assert id(janus.get_variabledef("SWING")) == id(referenceWingArea)
assert id(referenceWingArea.janus) == id(janus)

aeroBodyMomentCoefficient_Yaw.set_value(10)
print(aeroBodyMomentCoefficient_Yaw)